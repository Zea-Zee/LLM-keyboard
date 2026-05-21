from __future__ import annotations

from typing import Callable, Optional

from pynput import keyboard as pynput_keyboard

_MAX_BUFFER = 500


class TextBuffer:
    """Tracks typed text via a pynput global keyboard listener."""

    def __init__(self, on_space_callback: Callable[[str], None]) -> None:
        self._on_space = on_space_callback
        self._buf: list[str] = []
        self._listener: Optional[pynput_keyboard.Listener] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            suppress=False,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_context(self) -> str:
        return "".join(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    # ------------------------------------------------------------------
    def _on_press(self, key: pynput_keyboard.Key) -> None:
        try:
            if key == pynput_keyboard.Key.space:
                self._buf.append(" ")
                self._trim()
                self._on_space(self.get_context())

            elif key == pynput_keyboard.Key.backspace:
                if self._buf:
                    self._buf.pop()

            elif key in (
                pynput_keyboard.Key.enter,
                pynput_keyboard.Key.esc,
            ):
                self.clear()

            elif hasattr(key, "char") and key.char is not None:
                self._buf.append(key.char)
                self._trim()

            # Any other special key — ignore

        except Exception:
            pass

    def _trim(self) -> None:
        if len(self._buf) > _MAX_BUFFER:
            self._buf = self._buf[-_MAX_BUFFER:]
