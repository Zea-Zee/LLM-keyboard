from __future__ import annotations

from typing import List, Optional

import win32api
from pynput import keyboard as pynput_keyboard
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QWidget

from core.config import AppConfig
from core.inference_engine import InferenceEngine, InferenceWorker, ModelLoader
from core.text_buffer import TextBuffer

# ---------------------------------------------------------------------------
# Colour / style  (milky-white + mint/teal, compact)
# ---------------------------------------------------------------------------
_OVERLAY_H = 32           # ~1.5× smaller than original 44
_OVERLAY_MIN_W = 160      # 20% narrower (was 200)
_OVERLAY_MAX_W = 480      # 20% narrower (was 600)
_BG_COLOR = QColor(244, 250, 248, 235)   # milky-mint, slight transparency
_RADIUS = 8

_SS_WORD_BTN = (
    "QPushButton{"
    "background-color:rgba(221,243,239,210);"
    "color:#1D6B61; border:1px solid rgba(168,222,216,180);"
    "border-radius:5px;"
    "font-size:14px; padding:0 8px;}"
    "QPushButton:hover{background-color:rgba(168,222,216,230);}"
    "QPushButton:pressed{background-color:rgba(59,191,176,255); color:white;}"
)


class OverlayWindow(QWidget):
    """
    Frameless always-on-top overlay that shows word suggestions
    near the cursor position. Controlled by Ctrl+Shift+Space hotkey.
    """

    # Signals for cross-thread communication (pynput → Qt main thread)
    _suggestions_signal = pyqtSignal(list)
    _run_inference_signal = pyqtSignal(str)   # context string
    _toggle_signal = pyqtSignal()

    def __init__(
        self,
        engine: InferenceEngine,
        config: AppConfig,
    ) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.engine = engine
        self.config = config

        self._enabled: bool = True
        self._current_worker: Optional[InferenceWorker] = None
        self._kb_controller = pynput_keyboard.Controller()
        self._hotkeys: Optional[pynput_keyboard.GlobalHotKeys] = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedHeight(_OVERLAY_H)
        self.setMinimumWidth(_OVERLAY_MIN_W)
        self.setMaximumWidth(_OVERLAY_MAX_W)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(5, 2, 5, 2)
        self._layout.setSpacing(3)

        self._word_btns: List[QPushButton] = []

        self._suggestions_signal.connect(self._apply_suggestions)
        self._run_inference_signal.connect(self._start_inference)
        self._toggle_signal.connect(self._do_toggle)
        self._text_buf = TextBuffer(on_space_callback=self._on_space)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start model loading, pynput listener and hotkey handler."""
        loader = ModelLoader(self.engine)
        loader.loading_done.connect(self._text_buf.start)
        loader.loading_error.connect(self._on_load_error)
        loader.start()

        hotkey_str = self.config.hotkey_toggle
        self._hotkeys = pynput_keyboard.GlobalHotKeys(
            {hotkey_str: self._toggle_visibility}
        )
        self._hotkeys.start()

    def toggle_enabled(self) -> None:
        self._enabled = not self._enabled
        if not self._enabled:
            self.hide()

    # -----------------------------------------------------------------------
    # Painting (rounded + semi-transparent background)
    # -----------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(
            0, 0, self.width(), self.height(), _RADIUS, _RADIUS
        )
        painter.fillPath(path, _BG_COLOR)

    # -----------------------------------------------------------------------
    # Positioning
    # -----------------------------------------------------------------------

    def reposition(self) -> None:
        x, y = win32api.GetCursorPos()
        screen = QApplication.primaryScreen().geometry()
        half_w = self.width() // 2
        if y - 60 > 0:
            top = y - 54
        else:
            top = y + 20
        left = x - half_w
        self.move(left, top)
        self._clamp_to_screen(screen)

    def _clamp_to_screen(self, screen) -> None:
        x = max(screen.left(), min(self.x(), screen.right() - self.width()))
        y = max(screen.top(), min(self.y(), screen.bottom() - self.height()))
        self.move(x, y)

    # -----------------------------------------------------------------------
    # Suggestion slots
    # -----------------------------------------------------------------------

    def update_suggestions(self, words: List[str]) -> None:
        """Thread-safe update: emit signal to main thread."""
        self._suggestions_signal.emit(words)

    def _apply_suggestions(self, words: List[str]) -> None:
        nonempty = [w for w in words if w]
        if not nonempty:
            return

        # Resize width to fit words (96px per word after 20% width reduction)
        total_w = min(
            _OVERLAY_MAX_W,
            max(_OVERLAY_MIN_W, len(nonempty) * 96),
        )
        self.setFixedWidth(total_w)

        # Rebuild buttons
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._word_btns.clear()

        for word in nonempty:
            btn = QPushButton(word)
            btn.setStyleSheet(_SS_WORD_BTN)
            btn.setFixedHeight(_OVERLAY_H - 8)
            btn.clicked.connect(
                lambda _ch, w=word: self._on_word_clicked(w)
            )
            self._layout.addWidget(btn)
            self._word_btns.append(btn)

        if self._enabled:
            self.reposition()
            self.show()
            self.raise_()
            self.activateWindow()

    # -----------------------------------------------------------------------
    # Word click → type into active window
    # -----------------------------------------------------------------------

    def _on_word_clicked(self, word: str) -> None:
        self.hide()
        # Remove partial word that was already typed after last space
        ctx = self._text_buf.get_context()
        if ctx and not ctx.endswith(" "):
            parts = ctx.rsplit(" ", 1)
            partial = parts[-1] if len(parts) > 1 else ctx
            for _ in partial:
                self._kb_controller.press(pynput_keyboard.Key.backspace)
                self._kb_controller.release(pynput_keyboard.Key.backspace)
        self._kb_controller.type(word + " ")
        # Update buffer: drop partial word, append chosen word + space
        ctx = self._text_buf.get_context()
        if ctx and not ctx.endswith(" "):
            parts = ctx.rsplit(" ", 1)
            base = (parts[0] + " ") if len(parts) > 1 else ""
            self._text_buf.clear()
            for ch in (base + word + " "):
                self._text_buf._buf.append(ch)
        else:
            for ch in word + " ":
                self._text_buf._buf.append(ch)

    # -----------------------------------------------------------------------
    # Space callback (runs in pynput thread)
    # -----------------------------------------------------------------------

    def _on_space(self, context: str) -> None:
        """Called from pynput thread — must not touch Qt objects directly."""
        if not self._enabled:
            return
        if not self.engine.is_loaded():
            return
        if not context.strip():
            return
        # Emit signal → Qt queues call to _start_inference on main thread
        self._run_inference_signal.emit(context)

    @pyqtSlot(str)
    def _start_inference(self, context: str) -> None:
        worker = InferenceWorker(self.engine, context, k=5)
        worker.suggestions_ready.connect(self._apply_suggestions)
        worker.start()
        self._current_worker = worker

    # -----------------------------------------------------------------------
    # Hotkey toggle (pynput thread)
    # -----------------------------------------------------------------------

    def _toggle_visibility(self) -> None:
        """Called from pynput thread — emit signal to main thread."""
        self._toggle_signal.emit()

    @pyqtSlot()
    def _do_toggle(self) -> None:
        self._enabled = not self._enabled
        if not self._enabled:
            self.hide()

    # -----------------------------------------------------------------------
    # Error
    # -----------------------------------------------------------------------

    def _on_load_error(self, msg: str) -> None:
        pass  # overlay silently stays hidden on load failure

    # -----------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._text_buf.stop()
        if self._hotkeys:
            self._hotkeys.stop()
        super().closeEvent(event)
