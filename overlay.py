import sys

from PyQt6.QtWidgets import QApplication

from core.config import load_or_default
from core.inference_engine import InferenceEngine
from ui.overlay_window import OverlayWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    config = load_or_default()
    engine = InferenceEngine(config)
    window = OverlayWindow(engine, config)
    window.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
