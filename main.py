"""
LLM-Keyboard entry point.

Usage:
    python main.py              # keyboard mode (default)
    python main.py --keyboard   # keyboard mode
    python main.py --overlay    # overlay mode with system tray
    python main.py --settings   # open settings dialog only
"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from core.config import AppConfig, load_or_default
from core.inference_engine import InferenceEngine, ModelLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tray_icon() -> QIcon:
    """Generate a simple 16×16 coloured square as tray icon."""
    px = QPixmap(16, 16)
    px.fill(QColor("#0f3460"))
    return QIcon(px)


# ---------------------------------------------------------------------------
# Keyboard mode
# ---------------------------------------------------------------------------

def run_keyboard(config: AppConfig) -> int:
    from ui.keyboard_window import KeyboardWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LLM Keyboard")

    win = KeyboardWindow(config)
    win.show()

    return app.exec()


# ---------------------------------------------------------------------------
# Overlay mode
# ---------------------------------------------------------------------------

def run_overlay(config: AppConfig) -> int:
    from ui.overlay_window import OverlayWindow
    from ui.settings_window import SettingsWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LLM Keyboard")
    app.setQuitOnLastWindowClosed(False)

    engine = InferenceEngine(config)
    overlay = OverlayWindow(engine, config)
    overlay.start()

    # ── System tray ────────────────────────────────────────────────────────
    tray = QSystemTrayIcon(_make_tray_icon(), app)
    tray.setToolTip("LLM-Keyboard (Ctrl+Shift+Space)")

    menu = QMenu()

    toggle_action = menu.addAction("🔤 Подсказки: Вкл")
    _overlay_on = [True]

    def _toggle_overlay() -> None:
        _overlay_on[0] = not _overlay_on[0]
        overlay.toggle_enabled()
        label = "Вкл" if _overlay_on[0] else "Выкл"
        toggle_action.setText(f"🔤 Подсказки: {label}")

    toggle_action.triggered.connect(_toggle_overlay)
    menu.addSeparator()

    settings_action = menu.addAction("⚙ Настройки")

    def _open_settings() -> None:
        dlg = SettingsWindow(config)
        dlg.settings_changed.connect(
            lambda new_cfg: _reload_engine(new_cfg, overlay)
        )
        dlg.exec()

    settings_action.triggered.connect(_open_settings)

    training_action = menu.addAction("🏋 Обучение")

    def _open_training() -> None:
        try:
            from ui.training_window import TrainingWindow
            tw = TrainingWindow(config)
            tw.show()
        except ImportError:
            pass  # TrainingWindow not implemented yet

    training_action.triggered.connect(_open_training)

    menu.addSeparator()
    quit_action = menu.addAction("❌ Выход")
    quit_action.triggered.connect(app.quit)

    tray.setContextMenu(menu)
    tray.show()

    return app.exec()


def _reload_engine(new_config: AppConfig, overlay: object) -> None:
    overlay.engine.unload()
    overlay.engine = InferenceEngine(new_config)
    overlay.config = new_config
    loader = ModelLoader(overlay.engine)
    loader.loading_done.connect(overlay._text_buf.start)
    loader.start()


# ---------------------------------------------------------------------------
# Settings-only mode
# ---------------------------------------------------------------------------

def run_settings(config: AppConfig) -> int:
    from ui.settings_window import SettingsWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LLM Keyboard")

    dlg = SettingsWindow(config)
    dlg.exec()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-Keyboard")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--keyboard", action="store_true", help="Keyboard mode (default)"
    )
    group.add_argument(
        "--overlay", action="store_true", help="Overlay + system tray mode"
    )
    group.add_argument(
        "--settings", action="store_true", help="Open settings dialog"
    )
    args = parser.parse_args()

    config = load_or_default()

    if args.overlay:
        sys.exit(run_overlay(config))
    elif args.settings:
        sys.exit(run_settings(config))
    else:
        sys.exit(run_keyboard(config))


if __name__ == "__main__":
    main()
