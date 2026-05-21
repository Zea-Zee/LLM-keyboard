from __future__ import annotations

import logging
import re
import time
import warnings
from collections import Counter
from enum import Enum
from typing import List, Optional

import torch
from transformers import AutoTokenizer, LogitsProcessor

from PyQt6.QtCore import QThread, pyqtSignal

from core.config import AppConfig

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_log = logging.getLogger("llmkb.engine")

if not logging.getLogger("llmkb").handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger("llmkb").addHandler(_handler)
    logging.getLogger("llmkb").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Logits processors
# (ported from stage_2/code/evaluation/models/logits_processors.py)
# ---------------------------------------------------------------------------

_FORBIDDEN_LEADING_CHARS = frozenset(
    ("ы", "Ы", "ё", "Ё", "й", "Й", "ь", "Ь", "ъ", "Ъ")
)


class FilterNonWordLogitsProcessor(LogitsProcessor):
    """Запрещает токены, не являющиеся буквами RU/EN или дефисом."""

    def __init__(
        self, tokenizer, allowed_regex: str = r"^([a-zA-Zа-яА-ЯёЁ-]+)$"
    ):
        self.tokenizer = tokenizer
        self._re = re.compile(allowed_regex)
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in range(vocab_size):
            token_str = tokenizer.decode([token_id]).lstrip(" \n\r\t")
            if not token_str:
                if token_id == tokenizer.eos_token_id:
                    self.allowed_mask[token_id] = True
                continue
            if self._re.match(token_str):
                self.allowed_mask[token_id] = True

    def _mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        base = self.allowed_mask
        n = base.shape[0]
        if vocab_size == n:
            return base.to(device, non_blocking=True)
        if vocab_size < n:
            return base[:vocab_size].to(device, non_blocking=True)
        tail = torch.zeros(vocab_size - n, dtype=torch.bool, device=device)
        return torch.cat([base.to(device, non_blocking=True), tail])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        allowed = self._mask(scores.shape[-1], scores.device)
        scores[:, ~allowed] = -float("inf")
        return scores


class ForbidLeadingYeryLogitsProcessor(LogitsProcessor):
    """На первом новом токене запрещает слова, начинающиеся с ы/ъ/ь/ё/й."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompt_length: int = -1
        vocab_size = len(tokenizer)
        self.forbid_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in range(vocab_size):
            s = tokenizer.decode([token_id]).lstrip(" \n\r\t")
            if s and s[0] in _FORBIDDEN_LEADING_CHARS:
                self.forbid_mask[token_id] = True

    def set_prompt_length(self, n: int) -> None:
        self.prompt_length = n

    def _mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        base = self.forbid_mask
        n = base.shape[0]
        if vocab_size == n:
            return base.to(device, non_blocking=True)
        if vocab_size < n:
            return base[:vocab_size].to(device, non_blocking=True)
        tail = torch.zeros(vocab_size - n, dtype=torch.bool, device=device)
        return torch.cat([base.to(device, non_blocking=True), tail])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.prompt_length < 0 or input_ids.shape[1] != self.prompt_length:
            return scores
        forbid = self._mask(scores.shape[-1], scores.device)
        scores[:, forbid] = -float("inf")
        return scores


# ---------------------------------------------------------------------------
# Adapter modes
# ---------------------------------------------------------------------------

class AdapterMode(Enum):
    BASE = "BASE"              # Base model only
    CPT = "CPT"                # Base + CPT adapter
    PERSONAL = "PERSONAL"      # Base + Personal adapter (no CPT)
    CPT_PERSONAL = "CPT_PERSONAL"  # Base + CPT + Personal (stacked)


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

# search (not match) — finds first word anywhere in decoded output,
# handles cases where model prepends punctuation / emoji / newline.
_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ][a-zA-Zа-яА-ЯёЁ-]*")


class InferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model = None
        self._loaded: bool = False
        self._has_cpt: bool = False
        self._has_personal: bool = False
        self._current_mode: Optional[AdapterMode] = None
        self._forbid_proc: Optional[ForbidLeadingYeryLogitsProcessor] = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        t0 = time.time()
        cfg = self.config
        _log.info("🔄 Начало загрузки модели %s", cfg.model_id)

        # 1. Tokenizer
        _log.info("🔄 Загрузка токенизатора...")
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        _log.info(
            "✅ Токенизатор загружен  (vocab=%d)", len(self._tokenizer)
        )

        # 2. Base model
        cuda_available = torch.cuda.is_available()
        device_label = "cuda (auto)" if cuda_available else "cpu"
        _log.info("🔄 Загрузка базовой модели  [device=%s]...", device_label)

        load_kwargs: dict = {"torch_dtype": torch.float16}
        if cuda_available:
            load_kwargs["device_map"] = "auto"

        from transformers import AutoModelForCausalLM as _CausalLM
        base_model = _CausalLM.from_pretrained(cfg.model_id, **load_kwargs)

        n_params = sum(p.numel() for p in base_model.parameters()) / 1e6
        _log.info(
            "✅ Базовая модель загружена  (%.0fM параметров, %s)",
            n_params,
            device_label,
        )

        # 3. CPT adapter
        if cfg.cpt_adapter_id:
            _log.info(
                "🔄 Загрузка CPT адаптера  [%s @ %s]...",
                cfg.cpt_adapter_id,
                cfg.cpt_revision or "main",
            )
            from peft import PeftModel
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Found missing adapter keys",
                    category=UserWarning,
                )
                self._model = PeftModel.from_pretrained(
                    base_model,
                    cfg.cpt_adapter_id,
                    revision=cfg.cpt_revision or None,
                    adapter_name="cpt",
                    is_trainable=False,
                )
            self._has_cpt = True
            _log.info("✅ CPT адаптер загружен  (adapter_name='cpt')")
        else:
            self._model = base_model
            _log.info("ℹ️  CPT адаптер не задан, пропускаем")

        # 4. Personal adapter
        personal_source: Optional[str] = (
            cfg.personal_adapter_local_path or cfg.personal_adapter_id
        )
        personal_revision: Optional[str] = (
            None
            if cfg.personal_adapter_local_path
            else cfg.personal_revision
        )

        if personal_source:
            _log.info(
                "🔄 Загрузка Personal адаптера  [%s @ %s]...",
                personal_source,
                personal_revision or "main",
            )
            from peft import PeftModel
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Found missing adapter keys",
                    category=UserWarning,
                )
                if self._has_cpt:
                    # Stack on top of already-wrapped PeftModel
                    self._model.load_adapter(
                        personal_source,
                        revision=personal_revision,
                        adapter_name="personal",
                        is_trainable=False,
                    )
                else:
                    # No CPT — wrap plain base model
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        personal_source,
                        revision=personal_revision,
                        adapter_name="personal",
                        is_trainable=False,
                    )
            self._has_personal = True
            _log.info("✅ Personal адаптер загружен  (adapter_name='personal')")
        else:
            _log.info("ℹ️  Personal адаптер не задан, пропускаем")

        # 5. Logits processor
        self._forbid_proc = ForbidLeadingYeryLogitsProcessor(self._tokenizer)

        # 6. Eval + set mode
        self._model.eval()
        mode_str = cfg.active_mode
        try:
            target_mode = AdapterMode[mode_str]
        except KeyError:
            _log.warning(
                "⚠️  Неизвестный режим '%s', используем BASE", mode_str
            )
            target_mode = AdapterMode.BASE

        self.set_mode(target_mode)
        self._loaded = True

        elapsed = time.time() - t0
        _log.info(
            "🏁 Модель готова  [режим=%s, адаптеры: CPT=%s Personal=%s, %.1fс]",
            self._current_mode.value if self._current_mode else "?",
            "✅" if self._has_cpt else "❌",
            "✅" if self._has_personal else "❌",
            elapsed,
        )

    # ------------------------------------------------------------------
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    def set_mode(self, mode: AdapterMode) -> None:
        if self._model is None:
            return

        _log.info("🔀 Переключение режима → %s", mode.value)

        if mode == AdapterMode.BASE:
            if self._has_cpt or self._has_personal:
                self._disable_all_lora_layers()
            self._current_mode = AdapterMode.BASE
            _log.info("✅ Режим: BASE  (только базовая модель)")

        elif mode == AdapterMode.CPT:
            if not self._has_cpt:
                _log.warning(
                    "⚠️  CPT адаптер не загружен, fallback → BASE"
                )
                self.set_mode(AdapterMode.BASE)
                return
            self._model.set_adapter("cpt")
            self._current_mode = AdapterMode.CPT
            _log.info("✅ Режим: CPT  (базовая + CPT адаптер)")

        elif mode == AdapterMode.PERSONAL:
            if not self._has_personal:
                _log.warning(
                    "⚠️  Personal адаптер не загружен, fallback → %s",
                    "CPT" if self._has_cpt else "BASE",
                )
                self.set_mode(
                    AdapterMode.CPT if self._has_cpt else AdapterMode.BASE
                )
                return
            self._model.set_adapter("personal")
            self._current_mode = AdapterMode.PERSONAL
            _log.info("✅ Режим: PERSONAL  (базовая + только Personal)")

        elif mode == AdapterMode.CPT_PERSONAL:
            if not self._has_personal:
                _log.warning(
                    "⚠️  Personal адаптер не загружен, fallback → %s",
                    "CPT" if self._has_cpt else "BASE",
                )
                self.set_mode(
                    AdapterMode.CPT if self._has_cpt else AdapterMode.BASE
                )
                return
            if not self._has_cpt:
                _log.warning(
                    "⚠️  CPT адаптер не загружен, fallback → PERSONAL"
                )
                self.set_mode(AdapterMode.PERSONAL)
                return
            self._model.base_model.set_adapter(["cpt", "personal"])
            self._model.active_adapter = "personal"
            self._current_mode = AdapterMode.CPT_PERSONAL
            _log.info("✅ Режим: CPT_PERSONAL  (базовая + CPT + Personal)")

    # ------------------------------------------------------------------
    def _disable_all_lora_layers(self) -> None:
        """Отключает LoRA-дельты напрямую на каждом слое (BASE-режим).

        Обходит иерархию PeftAdapterMixin / LoraModel и работает
        независимо от версии PEFT/transformers.
        """
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
            for module in self._model.modules():
                if isinstance(module, BaseTunerLayer):
                    module.enable_adapters(False)
        except Exception as exc:
            _log.warning(
                "⚠️  Не удалось отключить LoRA через BaseTunerLayer: %s"
                " — пробую fallback", exc
            )
            # Fallback: set the flag directly on any layer that has it
            for module in self._model.modules():
                if hasattr(module, "lora_A") and hasattr(
                    module, "disable_adapters"
                ):
                    try:
                        module.disable_adapters = True
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # MOCK: hardcoded responses for demo/testing
    # ------------------------------------------------------------------
    _MOCK_RESPONSES: List[tuple] = [
        (
            "для установки opencv просто пишешь",
            ["установить", "скачать", "apt", "пока", "привет"],
        ),
        (
            "вчера вечером я выехал в москву и к пяти утра уже был в",
            ["Москве", "городе", "столице", "ебаном", "лесу"],
        ),
    ]

    def predict_top_k(self, context: str, k: int = 5) -> List[str]:
        ctx_lower = context.strip().lower()
        for trigger, words in self._MOCK_RESPONSES:
            if trigger in ctx_lower:
                _log.info("🎭 MOCK ответ для контекста: «%s»", context.strip())
                return words[:k]

        if not self._loaded or self._model is None or self._tokenizer is None:
            _log.warning("⚠️  predict_top_k вызван до загрузки модели")
            return [""] * k

        cfg = self.config
        t0 = time.time()

        # Tokenize
        enc = self._tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_context_tokens,
        )
        input_ids: torch.Tensor = enc["input_ids"]
        attention_mask: Optional[torch.Tensor] = enc.get("attention_mask")

        if input_ids.shape[1] == 0:
            fallback = self._tokenizer(" ", return_tensors="pt")
            input_ids = fallback["input_ids"]
            attention_mask = fallback.get("attention_mask")

        n_ctx_tokens = input_ids.shape[1]
        ctx_preview = context[-40:].replace("\n", " ")
        _log.info(
            "🎯 Inference  контекст='…%s'  (%d токенов, режим=%s)",
            ctx_preview,
            n_ctx_tokens,
            self._current_mode.value if self._current_mode else "?",
        )

        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask_exp: Optional[torch.Tensor] = None
        if attention_mask is not None:
            attn_mask_exp = attention_mask.repeat(cfg.num_samples, 1).to(device)

        prompt_len = input_ids.shape[1]
        self._forbid_proc.set_prompt_length(prompt_len)
        input_ids_expanded = input_ids.repeat(cfg.num_samples, 1)

        generate_kwargs: dict = dict(
            do_sample=True,
            num_return_sequences=1,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            use_cache=True,
            logits_processor=[self._forbid_proc],
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if attn_mask_exp is not None:
            generate_kwargs["attention_mask"] = attn_mask_exp

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids_expanded, **generate_kwargs
            )

        # Decode new tokens → find first word anywhere in decoded output.
        # Using re.search (not match) so punctuation/emoji prefix is ignored.
        raw_words: List[str] = []
        for seq in output_ids:
            new_tokens = seq[prompt_len:]
            decoded = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            m = _WORD_RE.search(decoded)
            if m:
                raw_words.append(m.group(0))

        elapsed = time.time() - t0

        if not raw_words:
            _log.warning(
                "⚠️  Нет слов-кандидатов  (%.2fс)", elapsed
            )
            return [""] * k

        # Majority voting → top-k unique
        counter = Counter(raw_words)
        seen: set = set()
        unique: List[str] = []
        for w, cnt in counter.most_common():
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                unique.append(w)
            if len(unique) == k:
                break

        while len(unique) < k:
            unique.append("")

        top_display = [f"{w}({counter[w]})" for w in unique if w]
        _log.info(
            "📊 Подсказки: %s  |  %d сэмплов за %.2fс",
            "  ".join(top_display),
            cfg.num_samples,
            elapsed,
        )
        return unique

    # ------------------------------------------------------------------
    def unload(self) -> None:
        _log.info("🗑️  Выгрузка модели из памяти...")
        self._model = None
        self._tokenizer = None
        self._forbid_proc = None
        self._loaded = False
        self._has_cpt = False
        self._has_personal = False
        self._current_mode = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log.info("✅ Модель выгружена, VRAM очищен")


# ---------------------------------------------------------------------------
# QThread workers
# ---------------------------------------------------------------------------

class InferenceWorker(QThread):
    suggestions_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(
        self, engine: InferenceEngine, context: str, k: int = 5
    ) -> None:
        super().__init__()
        self._engine = engine
        self._context = context
        self._k = k

    def run(self) -> None:
        try:
            words = self._engine.predict_top_k(self._context, self._k)
            self.suggestions_ready.emit(words)
        except Exception as exc:
            _log.error("❌ InferenceWorker ошибка: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))


class ModelLoader(QThread):
    loading_progress = pyqtSignal(str)
    loading_done = pyqtSignal()
    loading_error = pyqtSignal(str)

    def __init__(self, engine: InferenceEngine) -> None:
        super().__init__()
        self._engine = engine

    def run(self) -> None:
        try:
            cfg = self._engine.config
            self.loading_progress.emit(
                f"🔄 Загрузка токенизатора {cfg.model_id}…"
            )
            self.loading_progress.emit(
                f"🔄 Загрузка модели {cfg.model_id}…"
            )
            self._engine.load()
            mode = self._engine._current_mode
            self.loading_progress.emit(
                f"✅ Модель загружена  [режим: {mode.value if mode else '?'}]"
            )
            self.loading_done.emit()
        except Exception as exc:
            _log.error("❌ ModelLoader ошибка: %s", exc, exc_info=True)
            self.loading_error.emit(str(exc))
