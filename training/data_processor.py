"""
training/data_processor.py
Обёртка над parsing.telegram_parser для удобного использования из UI.

Пример использования:
    processor = TelegramDataProcessor()
    result = processor.process(
        export_folder="C:/TelegramExport",
        user_id="123456789",
        output_path="data/personal.jsonl",
    )
    token_count = processor.get_token_count("data/personal.jsonl")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from parsing.telegram_parser import parse_telegram_export

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Результат обработки
# ---------------------------------------------------------------------------

@dataclass
class ProcessResult:
    total_messages: int
    total_chunks: int
    total_tokens_estimate: int
    output_file: str
    success: bool
    error: Optional[str] = None

    @property
    def has_enough_tokens(self) -> bool:
        """Возвращает True если токенов достаточно для обучения (>= 10 000)."""
        return self.total_tokens_estimate >= 10_000


# ---------------------------------------------------------------------------
# Класс процессора
# ---------------------------------------------------------------------------

class TelegramDataProcessor:
    """
    Обрабатывает HTML-экспорт Telegram в JSONL-датасет.

    Использует parse_telegram_export() из parsing.telegram_parser.
    Добавляет валидацию, обработку ошибок и удобный dataclass результата.
    """

    def process(
        self,
        export_folder: str,
        user_id: str,
        output_path: str,
    ) -> ProcessResult:
        """
        Запустить полный pipeline: парсинг HTML → очистка → нарезка → JSONL.

        Args:
            export_folder: Папка с messages*.html файлами экспорта Telegram.
            user_id:       Числовой Telegram User ID в виде строки.
            output_path:   Путь для сохранения результирующего JSONL.

        Returns:
            ProcessResult с полями total_messages, total_chunks,
            total_tokens_estimate, output_file, success, error.
        """
        _log.info(
            "🚀 Запуск обработки экспорта Telegram  "
            "| папка=%s  user_id=%s",
            export_folder,
            user_id,
        )

        try:
            result = parse_telegram_export(
                export_folder=export_folder,
                user_id=user_id,
                output_path=output_path,
            )
        except FileNotFoundError as exc:
            _log.error("❌ Ошибка: %s", exc)
            return ProcessResult(
                total_messages=0,
                total_chunks=0,
                total_tokens_estimate=0,
                output_file="",
                success=False,
                error=str(exc),
            )
        except Exception as exc:  # pylint: disable=broad-except
            _log.error(
                "❌ Непредвиденная ошибка: %s", exc, exc_info=True
            )
            return ProcessResult(
                total_messages=0,
                total_chunks=0,
                total_tokens_estimate=0,
                output_file="",
                success=False,
                error=str(exc),
            )

        pr = ProcessResult(
            total_messages=result["total_messages"],    # type: ignore
            total_chunks=result["total_chunks"],        # type: ignore
            total_tokens_estimate=result[               # type: ignore
                "total_tokens_estimate"
            ],
            output_file=result["output_file"],          # type: ignore
            success=True,
        )

        if not pr.has_enough_tokens:
            _log.warning(
                "⚠️  Токенов маловато: ~%d (рекомендуется ≥10 000). "
                "Качество адаптера может быть низким.",
                pr.total_tokens_estimate,
            )
        else:
            _log.info(
                "✅ Обработка завершена: %d сообщений, %d чанков, ~%d токенов",
                pr.total_messages,
                pr.total_chunks,
                pr.total_tokens_estimate,
            )

        return pr

    # ------------------------------------------------------------------

    def get_token_count(self, jsonl_path: str) -> int:
        """
        Подсчитывает суммарное число символов в JSONL и возвращает
        оценку количества токенов (~0.6 токена на символ для рус. текста).

        Args:
            jsonl_path: Путь к JSONL файлу с полем "text".

        Returns:
            Целое число — оценка токенов.
        """
        path = Path(jsonl_path)
        if not path.is_file():
            _log.warning("⚠️  Файл не найден: %s", jsonl_path)
            return 0

        total_chars = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        total_chars += len(record.get("text", ""))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            _log.error("❌ Ошибка чтения файла %s: %s", jsonl_path, exc)
            return 0

        estimate = int(total_chars * 0.6)
        _log.info("🔢 Оценка токенов в %s: ~%d", path.name, estimate)
        return estimate

    def split_train_val(
        self,
        jsonl_path: str,
        val_ratio: float = 0.1,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Разбивает JSONL на train (90%) и val (10%) файлы.

        Args:
            jsonl_path:  Исходный JSONL.
            val_ratio:   Доля валидации (0.0 – 1.0, default 0.1).
            train_path:  Путь для train.jsonl (если None — рядом с исходным).
            val_path:    Путь для val.jsonl   (если None — рядом с исходным).

        Returns:
            Кортеж (train_path, val_path).
        """
        src = Path(jsonl_path)
        if not src.is_file():
            raise FileNotFoundError(f"Файл не найден: {jsonl_path}")

        with open(src, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]

        split_idx = max(1, int(len(lines) * (1.0 - val_ratio)))
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:]

        t_path = Path(train_path) if train_path else src.parent / "train.jsonl"
        v_path = Path(val_path) if val_path else src.parent / "val.jsonl"

        t_path.parent.mkdir(parents=True, exist_ok=True)
        v_path.parent.mkdir(parents=True, exist_ok=True)

        with open(t_path, "w", encoding="utf-8") as f:
            f.writelines(train_lines)
        with open(v_path, "w", encoding="utf-8") as f:
            f.writelines(val_lines)

        _log.info(
            "✂️  Split: train=%d чанков → %s | val=%d чанков → %s",
            len(train_lines),
            t_path.name,
            len(val_lines),
            v_path.name,
        )
        return str(t_path), str(v_path)
