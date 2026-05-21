"""
AppConfig — центральная конфигурация LLM-Keyboard.

Файл настроек хранится в:  ~/.llmkeyboard/settings.json
Редактировать можно:
  1. Через UI: кнопка ⚙ в клавиатурном окне → SettingsWindow
  2. Прямо в JSON-файле: %USERPROFILE%\\.llmkeyboard\\settings.json
  3. Через код:
        from core.config import load_or_default, get_settings_path
        cfg = load_or_default()
        cfg.num_samples = 30
        cfg.save(get_settings_path())
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    # ── Базовая модель ─────────────────────────────────────────────────────
    # HuggingFace repo id базовой модели.
    # Поддерживаются любые CausalLM совместимые с transformers.
    model_id: str = "Qwen/Qwen3.5-0.8B-Base"

    # ── CPT адаптер (LoRA дообученный на Telegram-корпусе) ────────────────
    # Чтобы отключить CPT адаптер — установить cpt_adapter_id = None.
    cpt_adapter_id: Optional[str] = "qzeaq/Qwen3.5-0.8B-telegram-qlora"
    # Ревизия (ветка / тег / commit hash) на HuggingFace Hub.
    cpt_revision: Optional[str] = "2.8kk-samples-cpt"

    # ── Personal адаптер (LoRA дообученный на личных сообщениях) ──────────
    # Чтобы отключить — установить personal_adapter_id = None.
    personal_adapter_id: Optional[str] = (
        "qzeaq/telegram-personal-adapters"
    )
    personal_revision: Optional[str] = "butilka-one-epoch"
    # Если задан local_path — используется вместо Hub (путь к папке адаптера).
    personal_adapter_local_path: Optional[str] = None

    # ── Хоткей overlay (формат pynput) ────────────────────────────────────
    # Примеры: "<ctrl>+<shift>+<space>", "<alt>+<space>", "<f12>"
    hotkey_toggle: str = "<ctrl>+<shift>+<space>"

    # ── Параметры генерации ───────────────────────────────────────────────
    # Сколько независимых сэмплов генерировать за один вызов.
    # Больше → лучше quality / diversity, но дольше.  Диапазон: 1–50.
    num_samples: int = 25

    # Максимум новых токенов на один сэмпл.
    # 3 → очень быстро, только первое слово;  6–8 → лучше покрытие.
    max_new_tokens: int = 6

    # Temperature: 0.1 = детерминировано, 2.0 = максимальная случайность.
    temperature: float = 0.7

    # top_k: рассматривать только топ-k токенов по вероятности.
    top_k: int = 50

    # top_p (nucleus sampling): сумма вероятностей не более p.
    top_p: float = 0.9

    # Максимальная длина контекста в токенах (усечение слева).
    max_context_tokens: int = 256

    # ── Устройство ────────────────────────────────────────────────────────
    # "auto" — cuda если доступен, иначе cpu.  Можно явно: "cuda", "cpu".
    device: str = "auto"

    # ── Режим адаптеров ───────────────────────────────────────────────────
    # Допустимые значения:
    #   "BASE"         — только базовая модель, без адаптеров
    #   "CPT"          — базовая + CPT адаптер
    #   "PERSONAL"     — базовая + только Personal адаптер (без CPT)
    #   "CPT_PERSONAL" — базовая + CPT + Personal (стек, рекомендуется)
    active_mode: str = "CPT_PERSONAL"

    # ── Обучение (используется TrainingWindow) ────────────────────────────
    # Путь к JSONL-файлу с тренировочными данными (выход парсера Telegram).
    training_data_path: Optional[str] = None
    # Куда сохранять обученный personal адаптер.
    training_output_path: str = "./adapters/personal"
    # Минимум токенов для запуска обучения (меньше → предупреждение).
    min_tokens_for_training: int = 10000

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )


def get_settings_path() -> Path:
    """Возвращает путь к settings.json (~/.llmkeyboard/settings.json)."""
    return Path.home() / ".llmkeyboard" / "settings.json"


def load_or_default() -> AppConfig:
    """Загружает конфиг из файла; при ошибке возвращает дефолтный."""
    path = get_settings_path()
    if path.exists():
        try:
            return AppConfig.load(path)
        except Exception:
            pass
    return AppConfig()
