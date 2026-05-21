"""
training/trainer.py
LoRA-обучение персонального адаптера поверх (опц.) CPT-адаптера.

Параметры оптимизированы под минимальное потребление VRAM:
  - load_in_8bit=True      (unsloth FastModel)
  - batch_size=1, grad_accum=8  → effective_batch=8
  - optim=paged_adamw_8bit  (экономичный оптимайзер)
  - gradient_checkpointing="unsloth"
"""
from __future__ import annotations

# ВАЖНО: установить ДО импорта unsloth
import os
os.environ.setdefault("UNSLOTH_DISABLE_FA2", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import builtins  # noqa: E402
import gc  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Callable, Optional  # noqa: E402

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Конфигурация обучения
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Параметры одного запуска LoRA-обучения."""

    # Данные
    train_path: str = ""
    val_path: str = ""
    output_dir: str = "./adapters/personal"

    # Модель
    model_id: str = "unsloth/Qwen3.5-0.8B-Base"
    cpt_adapter_id: Optional[str] = None
    cpt_revision: Optional[str] = None

    # Архитектура LoRA
    max_seq_length: int = 256
    lora_r: int = 4
    lora_alpha: int = 4          # обычно == lora_r

    # Оптимизация
    learning_rate: float = 5e-5
    batch_size: int = 1          # per_device; effective = batch_size * grad_accum
    grad_accum: int = 8          # effective_batch = 8
    num_epochs: int = 1
    early_stopping_patience: int = 4
    eval_steps: int = 50
    save_steps: int = 100

    # Временный чекпойнт (промежуточные шаги)
    checkpoint_dir: str = field(
        default_factory=lambda: os.path.join(
            os.environ.get("TEMP", "/tmp"),
            "llmkeyboard-checkpoints",
        )
    )


# ---------------------------------------------------------------------------
# Вспомогательные callbacks
# ---------------------------------------------------------------------------

class _StopCallback:
    """Прерывает тренинг когда выставлен stop_event."""

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop = stop_event

    # TrainerCallback-интерфейс (вызывается без наследования чтобы не
    # импортировать transformers на уровне модуля)
    def on_step_end(self, args, state, control, **kwargs):  # type: ignore
        if self._stop.is_set():
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore
        if self._stop.is_set():
            control.should_training_stop = True
        return control


class _MemoryCallback:
    """Освобождает CUDA-кэш после каждого шага."""

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


class _SaveBestCallback:
    """Сохраняет только персональный адаптер при улучшении eval_loss."""

    def __init__(
        self,
        output_dir: str,
        progress_cb: Callable[[str], None],
    ) -> None:
        self._dir = output_dir
        self._cb = progress_cb
        self._best = float("inf")
        self._model = None   # заполняется из LoRATrainer.train()
        self._tokenizer = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # type: ignore
        loss = metrics.get("eval_loss", float("inf"))
        if loss < self._best:
            self._best = loss
            if self._model is not None:
                self._model.save_pretrained(self._dir)
            if self._tokenizer is not None:
                self._tokenizer.save_pretrained(self._dir)
            self._cb(
                f"💾 Best adapter saved"
                f" (eval_loss={loss:.4f}) → {self._dir}"
            )
        return control


class _ProgressCallback:
    """Перенаправляет logging-события в progress_callback."""

    def __init__(self, cb: Callable[[str], None]) -> None:
        self._cb = cb

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
        if logs:
            step = state.global_step
            parts = [f"step={step}"]
            for k, v in logs.items():
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            self._cb("📊 " + " | ".join(parts))
        return control


# ---------------------------------------------------------------------------
# Перехват stdout/stderr → callback
# ---------------------------------------------------------------------------

class _CallbackStream(io.TextIOBase):
    """TextIO-обёртка: write() → callback."""

    def __init__(self, cb: Callable[[str], None]) -> None:
        super().__init__()
        self._cb = cb
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        if "\n" in self._buf:
            lines = self._buf.split("\n")
            for line in lines[:-1]:
                stripped = line.strip()
                if stripped:
                    self._cb(stripped)
            self._buf = lines[-1]
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._cb(self._buf.strip())
            self._buf = ""


# ---------------------------------------------------------------------------
# Главный класс
# ---------------------------------------------------------------------------

class LoRATrainer:
    """
    Обучает персональный LoRA-адаптер поверх базовой (+ опц. CPT) модели.

    Args:
        config:            Конфигурация обучения (TrainingConfig).
        progress_callback: Вызывается из потока обучения со строкой лога.
    """

    def __init__(
        self,
        config: TrainingConfig,
        progress_callback: Callable[[str], None],
    ) -> None:
        self.config = config
        self._cb = progress_callback
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Запустить обучение (блокирующий вызов).
        Вызывать из отдельного потока (QThread / threading.Thread).
        """
        self._stop_event.clear()
        self._cb("🚀 Начало обучения персонального адаптера")

        # Перехватываем stdout/stderr → callback
        _stream = _CallbackStream(self._cb)
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        _orig_print = builtins.print

        def _patched_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            self._cb(text)

        builtins.print = _patched_print  # type: ignore[assignment]
        sys.stdout = _stream  # type: ignore[assignment]
        sys.stderr = _stream  # type: ignore[assignment]

        try:
            self._run()
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
            builtins.print = _orig_print  # type: ignore[assignment]
            gc.collect()

    def stop(self) -> None:
        """Запросить досрочную остановку (потокобезопасно)."""
        self._stop_event.set()
        self._cb("⏹️  Получен запрос на остановку...")

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # Отложенный импорт — unsloth/transformers/trl могут быть не установлены
        try:
            from unsloth import FastModel
            from peft import PeftModel, LoraConfig
            from datasets import load_dataset
            from transformers import EarlyStoppingCallback
            from trl import SFTTrainer, SFTConfig
        except ImportError as exc:
            self._cb(f"❌ Ошибка импорта: {exc}")
            self._cb(
                "   Активируйте conda-окружение llmkeyboard "
                "и установите unsloth, trl, datasets."
            )
            raise

        cfg = self.config

        # ── 1. Загрузка базовой модели (8-bit) ──────────────────────────
        self._cb(f"⏳ Загрузка базовой модели {cfg.model_id} (8-bit)...")
        base_model, tokenizer = FastModel.from_pretrained(
            model_name=cfg.model_id,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,
            load_in_8bit=True,
        )
        self._cb("✅ Базовая модель загружена")

        if self._stop_event.is_set():
            self._cb("⏹️  Остановлено до начала обучения.")
            return

        # ── 2. CPT адаптер (замороженный) ────────────────────────────────
        if cfg.cpt_adapter_id:
            self._cb(
                f"⏳ Загрузка CPT адаптера "
                f"{cfg.cpt_adapter_id}@{cfg.cpt_revision}..."
            )
            model = PeftModel.from_pretrained(
                base_model,
                cfg.cpt_adapter_id,
                revision=cfg.cpt_revision,
                adapter_name="cpt",
                is_trainable=False,
            )
            for name, param in model.named_parameters():
                if "cpt" in name:
                    param.requires_grad = False
            self._cb("✅ CPT адаптер загружен и заморожен")

            # ── 3a. Персональный адаптер поверх CPT ─────────────────────
            personal_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter("personal", personal_config)
            model.set_adapter("personal")
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            self._cb("✅ Персональный адаптер добавлен поверх CPT")

        else:
            # ── 3b. Персональный адаптер без CPT ────────────────────────
            model = FastModel.get_peft_model(
                base_model,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules="all-linear",
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            self._cb("✅ Персональный адаптер добавлен (без CPT)")

        if self._stop_event.is_set():
            self._cb("⏹️  Остановлено до загрузки данных.")
            return

        # ── 4. Датасеты ──────────────────────────────────────────────────
        self._cb("📂 Загрузка датасетов...")
        train_dataset = load_dataset(
            "json", data_files=cfg.train_path, split="train"
        )
        val_dataset = load_dataset(
            "json", data_files=cfg.val_path, split="train"
        )
        train_dataset = train_dataset.select_columns(["text"])
        val_dataset = val_dataset.select_columns(["text"])
        self._cb(
            f"✅ Данные загружены: "
            f"train={len(train_dataset)}, val={len(val_dataset)} чанков"
        )

        # ── 5. Callbacks ─────────────────────────────────────────────────
        save_cb = _SaveBestCallback(
            output_dir=cfg.output_dir,
            progress_cb=self._cb,
        )
        save_cb._model = model
        save_cb._tokenizer = tokenizer

        stop_cb = _StopCallback(self._stop_event)
        mem_cb = _MemoryCallback()
        prog_cb = _ProgressCallback(self._cb)

        # Унифицируем под интерфейс TrainerCallback через обёртку
        from transformers import TrainerCallback

        class _Wrapper(TrainerCallback):
            def __init__(self, inner):
                self._inner = inner

            def on_step_end(self, args, state, control, **kw):
                return self._inner.on_step_end(args, state, control, **kw)

            def on_epoch_end(self, args, state, control, **kw):
                fn = getattr(self._inner, "on_epoch_end", None)
                if fn:
                    return fn(args, state, control, **kw)
                return control

            def on_evaluate(self, args, state, control, **kw):
                fn = getattr(self._inner, "on_evaluate", None)
                if fn:
                    return fn(args, state, control, **kw)
                return control

            def on_log(self, args, state, control, **kw):
                fn = getattr(self._inner, "on_log", None)
                if fn:
                    return fn(args, state, control, **kw)
                return control

        # ── 6. SFTConfig (memory-optimised) ──────────────────────────────
        total_steps = (
            (len(train_dataset) // cfg.batch_size)
            // cfg.grad_accum
            * cfg.num_epochs
        )
        _eval_steps = min(cfg.eval_steps, max(1, total_steps // 10))
        _save_steps = min(cfg.save_steps, max(1, total_steps // 5))

        training_args = SFTConfig(
            output_dir=cfg.checkpoint_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_steps=min(10, total_steps // 20),
            lr_scheduler_type="cosine",
            bf16=True,
            max_grad_norm=1.0,
            # --- экономичный оптимизатор ---
            optim="paged_adamw_8bit",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            # --- eval / save / log ---
            eval_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            eval_steps=_eval_steps,
            save_steps=_save_steps,
            logging_steps=max(1, _eval_steps // 2),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # --- SFT-специфика ---
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            packing=False,
            padding_free=False,
            eval_on_start=False,
            dataloader_drop_last=False,
            report_to="none",
        )

        self._cb(
            f"⚙️  Конфигурация: batch={cfg.batch_size}, "
            f"grad_accum={cfg.grad_accum} "
            f"(effective_batch={cfg.batch_size * cfg.grad_accum}), "
            f"lr={cfg.learning_rate}, epochs={cfg.num_epochs}, "
            f"~{total_steps} шагов"
        )

        # ── 7. Trainer ────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            callbacks=[
                _Wrapper(save_cb),
                _Wrapper(stop_cb),
                _Wrapper(mem_cb),
                _Wrapper(prog_cb),
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.early_stopping_patience
                ),
            ],
        )

        self._cb("🏋️  Обучение запущено...")
        trainer.train()

        if self._stop_event.is_set():
            self._cb("⏹️  Обучение остановлено пользователем.")
        else:
            self._cb(
                f"✅ Обучение завершено. "
                f"Лучший адаптер: {cfg.output_dir}"
            )
