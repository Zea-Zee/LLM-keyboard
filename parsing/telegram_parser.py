"""
parsing/telegram_parser.py
Парсинг HTML-экспорта Telegram в JSONL для последующего обучения LoRA-адаптера.

Использование:
    from parsing.telegram_parser import parse_telegram_export
    result = parse_telegram_export(
        export_folder="C:/TelegramExport",
        user_id="123456789",
        output_path="data/personal.jsonl",
    )
    # result: {"total_messages": N, "total_tokens_estimate": M, ...}

Формат экспорта Telegram Desktop (HTML):
    Файлы: messages.html, messages2.html, ...
    Структура: <div class="message default ...">
                   <div class="from_name">...</div>
                   <div class="text">...</div>
                   <div class="date" title="DD.MM.YYYY HH:MM:SS">
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
_SESSION_GAP = timedelta(minutes=60)   # пауза между сессиями
_CHUNK_SIZE = 200                       # символов в одном чанке
_CHUNK_OVERLAP = 100                    # перекрытие скользящего окна (50%)
_TOKENS_PER_CHAR = 0.6                  # оценка токенов для рус. текста

# Регулярки для очистки текста
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_RE_MULTI_WS = re.compile(r"\s{2,}")
_RE_FORWARDED = re.compile(r"^Forwarded from", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------
class _Message:
    __slots__ = ("from_id", "date", "text")

    def __init__(self, from_id: str, date: datetime, text: str) -> None:
        self.from_id = from_id
        self.date = date
        self.text = text


# ---------------------------------------------------------------------------
# Парсинг HTML
# ---------------------------------------------------------------------------

def _parse_date(raw: str) -> Optional[datetime]:
    """Парсит строку вида '20.05.2026 21:30:00' из атрибута title."""
    raw = raw.strip()
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_from_id(div: Tag) -> str:
    """Извлекает числовой ID отправителя из атрибута class div.message."""
    classes: List[str] = div.get("class", [])
    for cls in classes:
        # Telegram Desktop кодирует ID как 'peer_id_<числа>'
        m = re.match(r"from_id(\d+)", cls)
        if m:
            return m.group(1)
    # Альтернативный атрибут data-from-id
    from_id = div.get("data-from-id", "")
    return str(from_id).strip()


def _get_text_from_div(text_div: Tag) -> str:
    """Извлекает чистый текст из <div class='text'>."""
    # Удалить вложенные теги форматирования, оставив только текст
    text = text_div.get_text(separator=" ", strip=True)
    return text


def _clean_text(text: str) -> str:
    """Очищает текст: убирает ссылки, эмодзи, лишние пробелы."""
    text = _RE_URL.sub("", text)
    text = _RE_EMOJI.sub("", text)
    # Убрать форматирование markdown-подобных символов
    text = text.replace("\u200b", "").replace("\u00ad", "")
    text = _RE_MULTI_WS.sub(" ", text)
    return text.strip()


def _is_service_message(div: Tag) -> bool:
    """Возвращает True для системных/сервисных сообщений Telegram."""
    classes = div.get("class", [])
    return "service" in classes or "service_message" in classes


def _parse_html_file(path: Path) -> List[_Message]:
    """Парсит один HTML файл экспорта Telegram, возвращает список сообщений."""
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "lxml")

    messages: List[_Message] = []

    for div in soup.find_all("div", class_=re.compile(r"\bmessage\b")):
        if _is_service_message(div):
            continue

        # Дата
        date_div = div.find("div", class_="date")
        if not date_div:
            continue
        raw_date = date_div.get("title", "")
        date = _parse_date(raw_date)
        if date is None:
            continue

        # ID отправителя
        from_id = _extract_from_id(div)
        if not from_id:
            # Fallback: попробовать вложенный span с from_name
            from_name_div = div.find("div", class_="from_name")
            if from_name_div is None:
                continue
            from_id = ""  # не удалось определить — пропустим позже

        # Текст
        text_div = div.find("div", class_="text")
        if not text_div:
            continue

        raw_text = _get_text_from_div(text_div)  # type: ignore[arg-type]
        if _RE_FORWARDED.match(raw_text):
            continue

        text = _clean_text(raw_text)
        if not text:
            continue

        messages.append(_Message(from_id=from_id, date=date, text=text))

    return messages


# ---------------------------------------------------------------------------
# Сессионизация и нарезка
# ---------------------------------------------------------------------------

def _split_into_sessions(
    messages: List[_Message],
) -> List[List[_Message]]:
    """Разбивает отфильтрованные сообщения на сессии по паузам > 60 минут."""
    if not messages:
        return []

    sessions: List[List[_Message]] = []
    current: List[_Message] = [messages[0]]

    for prev, curr in zip(messages, messages[1:]):
        if curr.date - prev.date > _SESSION_GAP:
            sessions.append(current)
            current = []
        current.append(curr)

    if current:
        sessions.append(current)

    return sessions


def _session_to_text(session: List[_Message]) -> str:
    """Конкатенирует сообщения сессии в единую строку через перевод строки."""
    return "\n".join(msg.text for msg in session)


def _sliding_window_chunks(text: str) -> List[str]:
    """Нарезает текст на чанки ~200 символов (overlap=50%)."""
    if len(text) <= _CHUNK_SIZE:
        return [text] if text.strip() else []

    chunks: List[str] = []
    step = _CHUNK_SIZE - _CHUNK_OVERLAP  # 100 символов шаг

    start = 0
    while start < len(text):
        chunk = text[start: start + _CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def parse_telegram_export(
    export_folder: str,
    user_id: str,
    output_path: str,
) -> Dict[str, object]:
    """
    Парсит HTML-экспорт Telegram и записывает чанки в JSONL.

    Args:
        export_folder: Папка с файлами messages.html, messages2.html, ...
        user_id:       Числовой Telegram ID пользователя (строка).
        output_path:   Путь для записи результата (JSONL).

    Returns:
        {
            "total_messages":       int,   # отфильтрованных сообщений
            "total_chunks":         int,   # итоговых чанков
            "total_tokens_estimate": int,  # оценка токенов
            "output_file":          str,   # путь к JSONL
        }
    """
    folder = Path(export_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {export_folder}")

    # Найти все HTML файлы (messages.html, messages2.html, ...)
    html_files = sorted(
        folder.rglob("messages*.html"),
        key=lambda p: (p.parent, p.name),
    )
    if not html_files:
        raise FileNotFoundError(
            f"HTML файлы экспорта Telegram не найдены в: {export_folder}"
        )

    _log.info("📂 Найдено HTML файлов: %d", len(html_files))

    # Собрать все сообщения
    all_messages: List[_Message] = []
    for html_path in html_files:
        _log.info("🔍 Парсинг: %s", html_path.name)
        try:
            msgs = _parse_html_file(html_path)
            all_messages.extend(msgs)
            _log.info("   → %d сообщений", len(msgs))
        except Exception as exc:
            _log.warning("⚠️  Ошибка при парсинге %s: %s", html_path.name, exc)

    _log.info("📨 Всего сообщений в экспорте: %d", len(all_messages))

    # Отфильтровать по user_id
    user_messages = [m for m in all_messages if m.from_id == str(user_id)]
    _log.info(
        "👤 Сообщений пользователя (id=%s): %d",
        user_id,
        len(user_messages),
    )

    if not user_messages:
        _log.warning(
            "⚠️  Не найдено сообщений для user_id=%s. "
            "Проверьте ID (можно найти через @userinfobot в Telegram).",
            user_id,
        )

    # Отсортировать по времени
    user_messages.sort(key=lambda m: m.date)

    # Разбить на сессии
    sessions = _split_into_sessions(user_messages)
    _log.info("🗂️  Сессий (паузы >60 мин): %d", len(sessions))

    # Нарезать на чанки и записать в JSONL
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_chars = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for session in sessions:
            session_text = _session_to_text(session)
            chunks = _sliding_window_chunks(session_text)
            for chunk in chunks:
                record = {"text": chunk, "user_id": user_id}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
                total_chars += len(chunk)

    tokens_estimate = int(total_chars * _TOKENS_PER_CHAR)

    _log.info(
        "✅ JSONL записан: %s  |  чанков: %d  |  ~%d токенов",
        out_path,
        total_chunks,
        tokens_estimate,
    )

    return {
        "total_messages": len(user_messages),
        "total_chunks": total_chunks,
        "total_tokens_estimate": tokens_estimate,
        "output_file": str(out_path),
    }
