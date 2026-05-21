from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import AppConfig
from core.inference_engine import (
    AdapterMode,
    InferenceEngine,
    InferenceWorker,
    ModelLoader,
)

# ---------------------------------------------------------------------------
# Colour palette  (milky-white + mint/teal)
# ---------------------------------------------------------------------------
_BG     = "#F4FAF8"   # milky white with mint tint — window background
_BTN    = "#FFFFFF"   # white — key buttons
_ACCENT = "#3BBFB0"   # mint/teal — active, pressed, accents
_TEXT   = "#1D6B61"   # deep teal — all text
_HOVER  = "#DDF3EF"   # very light mint — hover state
_DARK   = "#E8F6F3"   # light mint — special keys background
_BORDER = "#A8DED8"   # soft teal — borders

# ---------------------------------------------------------------------------
# Keyboard layouts  (rows separated by list boundaries)
# ---------------------------------------------------------------------------
_LAYOUT_ORDER: List[str] = ["RU", "EN", "DIGITS", "SYMBOLS"]

_ROWS: Dict[str, List[List[str]]] = {
    "RU": [
        list("йцукенгшщзхъ"),
        list("фывапролджэ"),
        ["⇧"] + list("ячсмитьбю"),
    ],
    "EN": [
        list("qwertyuiop"),
        list("asdfghjkl"),
        ["⇧"] + list("zxcvbnm"),
    ],
    "DIGITS": [
        list("1234567890"),
        ["-"],
        list("()[]{}@#$%&*+=?!:;"),
    ],
    "SYMBOLS": [
        [".", ",", "'", '"', "<", ">", "/", "\\", "|"],
        ["~", "^", "`", "_", "…", "—", "«", "»", "№"],
        [],
    ],
}

# ---------------------------------------------------------------------------
# Stylesheets
# ---------------------------------------------------------------------------
_SS_WINDOW = f"background-color: {_BG}; color: {_TEXT};"

_SS_KEY = (
    f"QPushButton {{"
    f"background-color:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:5px; font-size:15px;}}"
    f"QPushButton:hover{{background-color:{_HOVER};}}"
    f"QPushButton:pressed{{background-color:{_ACCENT}; color:white;}}"
)

_SS_SPECIAL = (
    f"QPushButton {{"
    f"background-color:{_DARK}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:5px; font-size:13px;}}"
    f"QPushButton:hover{{background-color:{_HOVER};}}"
    f"QPushButton:pressed{{background-color:{_ACCENT}; color:white;}}"
)

_SS_SHIFT_ON = (
    f"QPushButton {{"
    f"background-color:{_ACCENT}; color:white;"
    f"border:1px solid {_ACCENT}; border-radius:5px; font-size:14px;}}"
)

_SS_SUGGEST = (
    f"QPushButton {{"
    f"background-color:{_DARK}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; font-size:13px;}}"
    f"QPushButton:hover{{background-color:{_HOVER};}}"
    f"QPushButton:pressed{{background-color:{_ACCENT}; color:white;}}"
    f"QPushButton:disabled{{background-color:{_BG}; color:{_BORDER};}}"
)

_SS_MODE_OFF = (
    f"QPushButton {{"
    f"background-color:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; font-size:12px;}}"
    f"QPushButton:hover{{background-color:{_HOVER};}}"
)

_SS_MODE_ON = (
    f"QPushButton {{"
    f"background-color:{_ACCENT}; color:white;"
    f"border:1px solid {_ACCENT}; border-radius:4px;"
    f"font-size:12px; font-weight:bold;}}"
)

_SS_TEXT_EDIT = (
    f"QTextEdit {{"
    f"background-color:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_ACCENT}; border-radius:6px;"
    f"padding:6px; font-size:15px;}}"
)

_SS_CLEAR = (
    f"QPushButton{{background-color:transparent; color:{_BORDER}; border:none;"
    f"font-size:16px;}}"
    f"QPushButton:hover{{color:{_TEXT};}}"
)

_WINDOW_W = 400
_WINDOW_H = 360


class KeyboardWindow(QWidget):
    """PyQt6 mobile-keyboard window (400 × 720 px, dark theme)."""

    def __init__(
        self,
        config: AppConfig,
        engine: Optional[InferenceEngine] = None,
    ) -> None:
        super().__init__()
        self.config = config
        if engine is None:
            engine = InferenceEngine(config)
        self.engine: InferenceEngine = engine

        self._layout_idx: int = 0
        self._shift_active: bool = False
        self._suggestion_btns: List[QPushButton] = []
        self._mode_btns: Dict[str, QPushButton] = {}
        self._current_worker: Optional[InferenceWorker] = None
        self._model_loader: Optional[ModelLoader] = None

        self._backspace_timer = QTimer(self)
        self._backspace_timer.setInterval(80)
        self._backspace_timer.timeout.connect(self._do_backspace)

        self._init_ui()
        self._rebuild_keyboard()
        self._show_placeholder_suggestions()
        self._refresh_mode_buttons()
        self._start_model_loading()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self) -> None:
        self.setWindowTitle("LLM Keyboard")
        self.setFixedSize(_WINDOW_W, _WINDOW_H)
        self.setStyleSheet(_SS_WINDOW)

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Text field + clear button ──────────────────────────────────────
        text_row = QWidget()
        text_row_layout = QHBoxLayout(text_row)
        text_row_layout.setContentsMargins(0, 0, 0, 0)
        text_row_layout.setSpacing(4)

        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText("Введите текст…")
        self._text_edit.setAcceptRichText(False)
        self._text_edit.setStyleSheet(_SS_TEXT_EDIT)
        self._text_edit.textChanged.connect(self._on_text_changed)
        self._update_text_height()
        text_row_layout.addWidget(self._text_edit)

        self._clear_btn = QPushButton("✕")
        self._clear_btn.setFixedSize(28, 28)
        self._clear_btn.setStyleSheet(_SS_CLEAR)
        self._clear_btn.setToolTip("Очистить")
        self._clear_btn.clicked.connect(self._clear_text)
        text_row_layout.addWidget(
            self._clear_btn,
            alignment=Qt.AlignmentFlag.AlignTop,
        )

        self._settings_btn = QPushButton("⚙")
        self._settings_btn.setFixedSize(28, 28)
        self._settings_btn.setStyleSheet(_SS_CLEAR)
        self._settings_btn.setToolTip("Настройки")
        self._settings_btn.clicked.connect(self._open_settings)
        text_row_layout.addWidget(
            self._settings_btn,
            alignment=Qt.AlignmentFlag.AlignTop,
        )

        root.addWidget(text_row)

        # ── Suggestions bar ────────────────────────────────────────────────
        self._suggestions_widget = QWidget()
        self._suggestions_layout = QHBoxLayout(self._suggestions_widget)
        self._suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self._suggestions_layout.setSpacing(4)
        self._suggestions_widget.setFixedHeight(44)
        root.addWidget(self._suggestions_widget)

        # ── Mode toggle buttons ────────────────────────────────────────────
        mode_bar = QWidget()
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        mode_bar.setFixedHeight(36)

        for label, mode_key in [
            ("Base", "BASE"),
            ("CPT", "CPT"),
            ("Personal", "PERSONAL"),
            ("CPT+P", "CPT_PERSONAL"),
        ]:
            btn = QPushButton(label)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            btn.clicked.connect(
                lambda _checked, m=mode_key: self._on_mode_changed(m)
            )
            mode_layout.addWidget(btn)
            self._mode_btns[mode_key] = btn

        root.addWidget(mode_bar)

        # ── Keyboard area (rebuilt dynamically) ────────────────────────────
        self._keyboard_widget = QWidget()
        self._keyboard_layout = QVBoxLayout(self._keyboard_widget)
        self._keyboard_layout.setContentsMargins(0, 0, 0, 0)
        self._keyboard_layout.setSpacing(3)
        root.addWidget(self._keyboard_widget, stretch=1)

    # -----------------------------------------------------------------------
    # Keyboard rebuild
    # -----------------------------------------------------------------------

    def _rebuild_keyboard(self) -> None:
        layout_name = _LAYOUT_ORDER[self._layout_idx]
        rows = _ROWS[layout_name]

        # Clear previous rows
        while self._keyboard_layout.count():
            item = self._keyboard_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        # Character rows
        for row_keys in rows:
            if not row_keys:
                continue
            row_w = self._make_key_row(row_keys, layout_name)
            self._keyboard_layout.addWidget(row_w, stretch=1)

        # Bottom control row
        self._keyboard_layout.addWidget(
            self._make_bottom_row(layout_name), stretch=1
        )

    def _make_key_row(
        self, keys: List[str], layout_name: str
    ) -> QWidget:
        row_w = QWidget()
        hl = QHBoxLayout(row_w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(3)

        for key in keys:
            if key == "⇧":
                btn = self._make_shift_btn()
                hl.addWidget(btn, stretch=2)
            else:
                display = key.upper() if self._shift_active else key
                btn = QPushButton(display)
                btn.setStyleSheet(_SS_KEY)
                btn.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Expanding,
                )
                # font size smaller for dense rows (DIGITS row3, SYMBOLS)
                if layout_name in ("DIGITS", "SYMBOLS") and len(keys) > 10:
                    font = btn.font()
                    font.setPointSize(11)
                    btn.setFont(font)
                char = key
                btn.pressed.connect(
                    lambda c=char: self._on_char_pressed(c)
                )
                hl.addWidget(btn, stretch=1)

        return row_w

    def _make_shift_btn(self) -> QPushButton:
        btn = QPushButton("⇧")
        btn.setStyleSheet(
            _SS_SHIFT_ON if self._shift_active else _SS_SPECIAL
        )
        btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        btn.clicked.connect(self._toggle_shift)
        return btn

    def _make_bottom_row(self, layout_name: str) -> QWidget:
        row_w = QWidget()
        hl = QHBoxLayout(row_w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(3)

        lang_btn = QPushButton(layout_name)
        lang_btn.setStyleSheet(_SS_SPECIAL)
        lang_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lang_btn.clicked.connect(self._cycle_layout)
        hl.addWidget(lang_btn, stretch=1)

        comma_btn = QPushButton(",")
        comma_btn.setStyleSheet(_SS_KEY)
        comma_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        comma_btn.pressed.connect(lambda: self._on_char_pressed(","))
        hl.addWidget(comma_btn, stretch=1)

        space_btn = QPushButton("ПРОБЕЛ")
        space_btn.setStyleSheet(_SS_SPECIAL)
        space_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        space_btn.pressed.connect(self._on_space_pressed)
        hl.addWidget(space_btn, stretch=3)

        dot_btn = QPushButton(".")
        dot_btn.setStyleSheet(_SS_KEY)
        dot_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        dot_btn.pressed.connect(lambda: self._on_char_pressed("."))
        hl.addWidget(dot_btn, stretch=1)

        bs_btn = QPushButton("⌫")
        bs_btn.setStyleSheet(_SS_SPECIAL)
        bs_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        bs_btn.pressed.connect(self._start_backspace)
        bs_btn.released.connect(self._stop_backspace)
        hl.addWidget(bs_btn, stretch=1)

        return row_w

    # -----------------------------------------------------------------------
    # Key event handlers
    # -----------------------------------------------------------------------

    def _on_char_pressed(self, char: str) -> None:
        text = char.upper() if self._shift_active else char
        self._text_edit.insertPlainText(text)
        if self._shift_active:
            self._shift_active = False
            self._rebuild_keyboard()

    def _on_space_pressed(self) -> None:
        self._text_edit.insertPlainText(" ")
        context = self._text_edit.toPlainText()
        if context.strip():
            self._run_inference(context)

    def _toggle_shift(self) -> None:
        self._shift_active = not self._shift_active
        self._rebuild_keyboard()

    def _cycle_layout(self) -> None:
        self._layout_idx = (self._layout_idx + 1) % len(_LAYOUT_ORDER)
        self._shift_active = False
        self._rebuild_keyboard()

    def _start_backspace(self) -> None:
        self._do_backspace()
        self._backspace_timer.start()

    def _stop_backspace(self) -> None:
        self._backspace_timer.stop()

    def _do_backspace(self) -> None:
        cursor = self._text_edit.textCursor()
        cursor.deletePreviousChar()
        self._text_edit.setTextCursor(cursor)

    def _clear_text(self) -> None:
        self._text_edit.clear()

    # -----------------------------------------------------------------------
    # Mode buttons
    # -----------------------------------------------------------------------

    def _on_mode_changed(self, mode_key: str) -> None:
        self.config.active_mode = mode_key
        if self.engine and self.engine.is_loaded():
            self.engine.set_mode(AdapterMode[mode_key])
            context = self._text_edit.toPlainText()
            if context.strip():
                self._run_inference(context)
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self) -> None:
        for key, btn in self._mode_btns.items():
            active = key == self.config.active_mode
            btn.setStyleSheet(_SS_MODE_ON if active else _SS_MODE_OFF)

    # -----------------------------------------------------------------------
    # Inference stubs (fully wired in Step 4)
    # -----------------------------------------------------------------------

    def _run_inference(self, context: str) -> None:
        if self.engine is None or not self.engine.is_loaded():
            return
        self._show_loading_suggestions()
        worker = InferenceWorker(self.engine, context, k=5)
        worker.suggestions_ready.connect(self._update_suggestions)
        worker.error_occurred.connect(self._on_inference_error)
        worker.start()
        self._current_worker = worker

    def _on_inference_error(self, msg: str) -> None:
        self._show_placeholder_suggestions()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _start_model_loading(self) -> None:
        self._model_loader = ModelLoader(self.engine)
        self._model_loader.loading_progress.connect(
            self._on_loading_progress
        )
        self._model_loader.loading_done.connect(self._on_loading_done)
        self._model_loader.loading_error.connect(self._on_loading_error)
        self._model_loader.start()

    def _on_loading_progress(self, msg: str) -> None:
        self._set_suggestion_slots(
            [msg], n=1, font_size=10, enabled=False
        )

    def _on_loading_done(self) -> None:
        context = self._text_edit.toPlainText()
        if context.strip():
            self._run_inference(context)
        else:
            self._show_placeholder_suggestions()

    def _on_loading_error(self, msg: str) -> None:
        QMessageBox.critical(
            self,
            "Ошибка загрузки модели",
            f"Не удалось загрузить модель:\n\n{msg}",
        )
        self._show_placeholder_suggestions()

    # -----------------------------------------------------------------------
    # Settings (stub — fully connected in Step 6)
    # -----------------------------------------------------------------------

    def _open_settings(self) -> None:
        from ui.settings_window import SettingsWindow
        dlg = SettingsWindow(self.config, parent=self)
        dlg.settings_changed.connect(self._on_settings_changed)
        dlg.exec()

    def _on_settings_changed(self, new_config: AppConfig) -> None:
        self.config = new_config
        self.engine.unload()
        self.engine = InferenceEngine(new_config)
        self._refresh_mode_buttons()
        self._start_model_loading()

    # -----------------------------------------------------------------------
    # Suggestions
    # -----------------------------------------------------------------------

    def _show_placeholder_suggestions(self) -> None:
        engine_ready = self.engine is not None and self.engine.is_loaded()
        label = "⏳ Загрузка модели…" if not engine_ready else ""
        self._set_suggestion_slots(
            [label] + [""] * 4, n=1 if label else 3, font_size=11,
            enabled=False,
        )

    def _show_loading_suggestions(self) -> None:
        self._set_suggestion_slots(
            ["…", "…", "…", "…", "…"], n=5, font_size=12, enabled=False
        )

    def _update_suggestions(self, words: List[str]) -> None:
        if not any(w for w in words):
            self._show_placeholder_suggestions()
            return
        n, fs = self._compute_slots(words)
        self._set_suggestion_slots(
            words[:n] + [""] * max(0, n - len(words)),
            n=n, font_size=fs, enabled=True,
        )

    def _compute_slots(self, words: List[str]) -> tuple[int, int]:
        """Return (num_slots, font_pt) for the adaptive suggestions bar."""
        total_w = self.width() - 20
        max_fs, min_fs = 14, 9
        for n in [5, 4, 3]:
            slot_w = total_w // n
            test = [w for w in words[:n] if w]
            for fs in range(max_fs, min_fs - 1, -1):
                font = QFont()
                font.setPointSize(fs)
                fm = QFontMetrics(font)
                fits = all(
                    fm.horizontalAdvance(w) + 16 <= slot_w for w in test
                )
                if fits:
                    return n, fs
        return 3, min_fs

    def _set_suggestion_slots(
        self,
        words: List[str],
        n: int,
        font_size: int,
        enabled: bool,
    ) -> None:
        # Remove old buttons
        while self._suggestions_layout.count():
            item = self._suggestions_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._suggestion_btns.clear()

        font = QFont()
        font.setPointSize(font_size)

        for i in range(n):
            word = words[i] if i < len(words) else ""
            btn = QPushButton(word)
            btn.setStyleSheet(_SS_SUGGEST)
            btn.setFont(font)
            btn.setEnabled(enabled and bool(word))
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            if word and enabled:
                btn.clicked.connect(
                    lambda _ch, w=word: self._on_suggestion_clicked(w)
                )
            self._suggestions_layout.addWidget(btn)
            self._suggestion_btns.append(btn)

    def _on_suggestion_clicked(self, word: str) -> None:
        text = self._text_edit.toPlainText()
        # Replace trailing partial word with suggestion
        if text and not text.endswith(" "):
            # strip last word (incomplete)
            parts = text.rsplit(" ", 1)
            base = parts[0] + " " if len(parts) > 1 else ""
            self._text_edit.setPlainText(base + word + " ")
        else:
            self._text_edit.insertPlainText(word + " ")
        # Move cursor to end
        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
        # Re-run inference
        context = self._text_edit.toPlainText()
        if context.strip():
            self._run_inference(context)

    # -----------------------------------------------------------------------
    # Text field auto-expand
    # -----------------------------------------------------------------------

    def _on_text_changed(self) -> None:
        self._update_text_height()

    def _update_text_height(self) -> None:
        doc = self._text_edit.document()
        doc.setTextWidth(self._text_edit.viewport().width())
        content_h = int(doc.size().height())
        fm = self._text_edit.fontMetrics()
        line_h = fm.lineSpacing()
        min_h = line_h + 16
        max_h = line_h * 3 + 16
        self._text_edit.setFixedHeight(
            max(min_h, min(content_h + 16, max_h))
        )
