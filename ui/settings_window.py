from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import AppConfig, get_settings_path

# ---------------------------------------------------------------------------
# Palette (milky-white + mint/teal, matches keyboard window)
# ---------------------------------------------------------------------------
_BG      = "#F4FAF8"
_BTN     = "#FFFFFF"
_ACCENT  = "#3BBFB0"
_TEXT    = "#1D6B61"
_HOVER   = "#DDF3EF"
_DARK    = "#E8F6F3"
_BORDER  = "#A8DED8"
_INPUT_BG = "#FFFFFF"

_SS_DIALOG = (
    f"QDialog{{background:{_BG}; color:{_TEXT};}}"
    f"QGroupBox{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:6px;"
    f"margin-top:8px; font-weight:bold;}}"
    f"QGroupBox::title{{subcontrol-origin:margin;"
    f"subcontrol-position:top left; padding:0 6px;}}"
    f"QLabel{{color:{_TEXT};}}"
    f"QCheckBox{{color:{_TEXT};}}"
    f"QRadioButton{{color:{_TEXT};}}"
    f"QLineEdit{{background:{_INPUT_BG}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; padding:4px;}}"
    f"QSpinBox,QDoubleSpinBox{{background:{_INPUT_BG}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; padding:2px;}}"
    f"QSlider::groove:horizontal{{background:{_DARK}; height:4px;"
    f"border-radius:2px;}}"
    f"QSlider::handle:horizontal{{background:{_ACCENT}; width:14px;"
    f"height:14px; margin:-5px 0; border-radius:7px;}}"
    f"QSlider::sub-page:horizontal{{background:{_ACCENT};"
    f"border-radius:2px;}}"
    f"QPushButton{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px;"
    f"padding:5px 12px;}}"
    f"QPushButton:hover{{background:{_HOVER};}}"
    f"QPushButton:pressed{{background:{_ACCENT}; color:white;}}"
    f"QScrollArea{{background:{_BG}; border:none;}}"
    f"QWidget#scroll_content{{background:{_BG};}}"
)

_SS_SAVE_RELOAD = (
    f"QPushButton{{background:{_ACCENT}; color:white;"
    f"border:1px solid {_ACCENT}; border-radius:4px;"
    f"padding:5px 12px; font-weight:bold;}}"
    f"QPushButton:hover{{background:#2EA898;}}"
)


class _AdapterSection(QGroupBox):
    """Reusable group for one adapter (CPT or Personal)."""

    def __init__(self, title: str, use_label: str, parent=None) -> None:
        super().__init__(title, parent)
        self._build(use_label)

    def _build(self, use_label: str) -> None:
        vl = QVBoxLayout(self)
        vl.setSpacing(6)

        self.chk_use = QCheckBox(use_label)
        self.chk_use.setChecked(True)
        vl.addWidget(self.chk_use)

        # Hub / Local radio
        radio_row = QHBoxLayout()
        self.rb_hub = QRadioButton("HuggingFace Hub")
        self.rb_local = QRadioButton("Локальная папка")
        self.rb_hub.setChecked(True)
        radio_row.addWidget(self.rb_hub)
        radio_row.addWidget(self.rb_local)
        radio_row.addStretch()
        vl.addLayout(radio_row)

        # Hub fields
        self._hub_widget = QWidget()
        hub_form = QFormLayout(self._hub_widget)
        hub_form.setContentsMargins(0, 0, 0, 0)
        self.le_repo = QLineEdit()
        self.le_repo.setPlaceholderText("user/repo-name")
        self.le_revision = QLineEdit()
        self.le_revision.setPlaceholderText("main")
        hub_form.addRow("Repo ID:", self.le_repo)
        hub_form.addRow("Revision:", self.le_revision)
        vl.addWidget(self._hub_widget)

        # Local fields
        self._local_widget = QWidget()
        local_row = QHBoxLayout(self._local_widget)
        local_row.setContentsMargins(0, 0, 0, 0)
        self.le_path = QLineEdit()
        self.le_path.setPlaceholderText("Путь к папке адаптера…")
        self._browse_btn = QPushButton("Обзор…")
        self._browse_btn.setFixedWidth(80)
        self._browse_btn.clicked.connect(self._browse)
        local_row.addWidget(self.le_path)
        local_row.addWidget(self._browse_btn)
        self._local_widget.setVisible(False)
        vl.addWidget(self._local_widget)

        # Wire radio toggles
        self.rb_hub.toggled.connect(self._on_radio)
        self.chk_use.toggled.connect(self._on_use_toggled)

    def _on_radio(self, hub_checked: bool) -> None:
        self._hub_widget.setVisible(hub_checked)
        self._local_widget.setVisible(not hub_checked)

    def _on_use_toggled(self, checked: bool) -> None:
        self.rb_hub.setEnabled(checked)
        self.rb_local.setEnabled(checked)
        self._hub_widget.setEnabled(checked)
        self._local_widget.setEnabled(checked)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку адаптера"
        )
        if folder:
            self.le_path.setText(folder)

    # --- serialisation helpers -------------------------------------------

    def load_from_config(
        self,
        adapter_id: str | None,
        revision: str | None,
        local_path: str | None,
    ) -> None:
        if local_path:
            self.rb_local.setChecked(True)
            self.le_path.setText(local_path)
        else:
            self.rb_hub.setChecked(True)
            self.le_repo.setText(adapter_id or "")
            self.le_revision.setText(revision or "")
        self.chk_use.setChecked(
            bool(adapter_id or local_path)
        )

    @property
    def is_enabled(self) -> bool:
        return self.chk_use.isChecked()

    @property
    def hub_id(self) -> Optional[str]:
        v = self.le_repo.text().strip()
        return v or None

    @property
    def hub_revision(self) -> Optional[str]:
        v = self.le_revision.text().strip()
        return v or None

    @property
    def local_path(self) -> Optional[str]:
        v = self.le_path.text().strip()
        return v or None

    def use_local(self) -> bool:
        return self.rb_local.isChecked()


class SettingsWindow(QDialog):
    """Settings dialog for adapters and generation parameters."""

    settings_changed = pyqtSignal(AppConfig)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._capturing_hotkey = False
        self._capture_timer: Optional[QTimer] = None

        self.setWindowTitle("Настройки LLM Keyboard")
        self.setMinimumWidth(480)
        self.setStyleSheet(_SS_DIALOG)

        self._build_ui()
        self._load_config()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("scroll_content")
        content_vl = QVBoxLayout(content)
        content_vl.setSpacing(10)
        content_vl.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        # ── CPT Adapter ───────────────────────────────────────────────────
        self._cpt = _AdapterSection("CPT Адаптер", "Использовать CPT адаптер")
        content_vl.addWidget(self._cpt)

        # ── Personal Adapter ──────────────────────────────────────────────
        self._personal = _AdapterSection(
            "Персональный Адаптер", "Использовать персональный адаптер"
        )
        content_vl.addWidget(self._personal)

        # ── Generation params ─────────────────────────────────────────────
        gen_group = QGroupBox("Параметры генерации")
        gen_form = QFormLayout(gen_group)
        gen_form.setSpacing(8)

        self._samples_slider, self._samples_spin = self._make_int_slider(
            1, 30, 10
        )
        gen_form.addRow(
            "Число сэмплов:",
            self._slider_row(self._samples_slider, self._samples_spin),
        )

        self._temp_slider, self._temp_spin = self._make_float_slider(
            0.1, 2.0, 0.7
        )
        gen_form.addRow(
            "Temperature:",
            self._slider_row(self._temp_slider, self._temp_spin),
        )

        self._ctx_spin = QSpinBox()
        self._ctx_spin.setRange(64, 512)
        self._ctx_spin.setValue(256)
        self._ctx_spin.setSuffix(" токенов")
        gen_form.addRow("Макс. контекст:", self._ctx_spin)

        content_vl.addWidget(gen_group)

        # ── Hotkey ────────────────────────────────────────────────────────
        hotkey_group = QGroupBox("Хоткей overlay")
        hotkey_vl = QVBoxLayout(hotkey_group)

        hotkey_row = QHBoxLayout()
        self._hotkey_edit = QLineEdit()
        self._hotkey_edit.setReadOnly(True)
        self._capture_btn = QPushButton("Изменить")
        self._capture_btn.setFixedWidth(90)
        self._capture_btn.clicked.connect(self._start_capture)
        hotkey_row.addWidget(self._hotkey_edit)
        hotkey_row.addWidget(self._capture_btn)
        hotkey_vl.addLayout(hotkey_row)

        self._capture_label = QLabel("")
        self._capture_label.setStyleSheet("color:#aaa; font-size:11px;")
        hotkey_vl.addWidget(self._capture_label)

        content_vl.addWidget(hotkey_group)
        content_vl.addStretch()

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._save_reload_btn = QPushButton("Сохранить и перезагрузить модель")
        self._save_reload_btn.setStyleSheet(_SS_SAVE_RELOAD)
        self._save_reload_btn.clicked.connect(self._on_save_reload)

        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self._on_save)

        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(self._save_reload_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    # -----------------------------------------------------------------------
    # Slider helpers
    # -----------------------------------------------------------------------

    def _make_int_slider(
        self, lo: int, hi: int, val: int
    ) -> tuple[QSlider, QSpinBox]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(val)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setFixedWidth(64)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return slider, spin

    def _make_float_slider(
        self, lo: float, hi: float, val: float
    ) -> tuple[QSlider, QDoubleSpinBox]:
        factor = 10
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(lo * factor), int(hi * factor))
        slider.setValue(int(val * factor))
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setValue(val)
        spin.setFixedWidth(72)
        slider.valueChanged.connect(lambda v: spin.setValue(v / factor))
        spin.valueChanged.connect(
            lambda v: slider.setValue(int(v * factor))
        )
        return slider, spin

    @staticmethod
    def _slider_row(slider: QSlider, spin: QWidget) -> QWidget:
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(slider)
        hl.addWidget(spin)
        return w

    # -----------------------------------------------------------------------
    # Hotkey capture
    # -----------------------------------------------------------------------

    def _start_capture(self) -> None:
        self._capturing_hotkey = True
        self._pressed_keys: list[str] = []
        self._capture_label.setText("Нажмите комбинацию клавиш…")
        self._capture_btn.setEnabled(False)

        from pynput import keyboard as pk

        def on_press(key):
            name = self._key_name(key)
            if name and name not in self._pressed_keys:
                self._pressed_keys.append(name)

        def on_release(key):
            if not self._capturing_hotkey:
                return False
            if len(self._pressed_keys) >= 2:
                combo = "+".join(self._pressed_keys)
                self._hotkey_edit.setText(combo)
                self._capture_label.setText("")
                self._capture_btn.setEnabled(True)
                self._capturing_hotkey = False
                return False

        self._pynput_listener = pk.Listener(
            on_press=on_press, on_release=on_release
        )
        self._pynput_listener.start()

    @staticmethod
    def _key_name(key) -> Optional[str]:
        from pynput import keyboard as pk
        mapping = {
            pk.Key.ctrl_l: "<ctrl>",
            pk.Key.ctrl_r: "<ctrl>",
            pk.Key.shift_l: "<shift>",
            pk.Key.shift_r: "<shift>",
            pk.Key.alt_l: "<alt>",
            pk.Key.alt_r: "<alt>",
            pk.Key.space: "<space>",
        }
        if key in mapping:
            return mapping[key]
        if hasattr(key, "char") and key.char:
            return key.char
        return None

    # -----------------------------------------------------------------------
    # Load / save config
    # -----------------------------------------------------------------------

    def _load_config(self) -> None:
        cfg = self.config
        self._cpt.load_from_config(
            cfg.cpt_adapter_id, cfg.cpt_revision, None
        )
        self._personal.load_from_config(
            cfg.personal_adapter_id,
            cfg.personal_revision,
            cfg.personal_adapter_local_path,
        )
        self._samples_spin.setValue(cfg.num_samples)
        self._temp_spin.setValue(cfg.temperature)
        self._ctx_spin.setValue(cfg.max_context_tokens)
        self._hotkey_edit.setText(cfg.hotkey_toggle)

    def _collect_config(self) -> AppConfig:
        cfg = AppConfig(
            model_id=self.config.model_id,
            cpt_adapter_id=(
                self._cpt.hub_id
                if self._cpt.is_enabled and not self._cpt.use_local()
                else self.config.cpt_adapter_id
            ),
            cpt_revision=(
                self._cpt.hub_revision
                if self._cpt.is_enabled and not self._cpt.use_local()
                else self.config.cpt_revision
            ),
            personal_adapter_id=(
                self._personal.hub_id
                if self._personal.is_enabled and not self._personal.use_local()
                else None
            ),
            personal_revision=(
                self._personal.hub_revision
                if self._personal.is_enabled
                and not self._personal.use_local()
                else None
            ),
            personal_adapter_local_path=(
                self._personal.local_path
                if self._personal.is_enabled and self._personal.use_local()
                else None
            ),
            hotkey_toggle=self._hotkey_edit.text().strip()
            or self.config.hotkey_toggle,
            num_samples=self._samples_spin.value(),
            temperature=self._temp_spin.value(),
            max_context_tokens=self._ctx_spin.value(),
            device=self.config.device,
            active_mode=self.config.active_mode,
            training_data_path=self.config.training_data_path,
            training_output_path=self.config.training_output_path,
            min_tokens_for_training=self.config.min_tokens_for_training,
        )
        return cfg

    def _on_save(self) -> None:
        cfg = self._collect_config()
        cfg.save(get_settings_path())
        self.config = cfg
        self.accept()

    def _on_save_reload(self) -> None:
        cfg = self._collect_config()
        cfg.save(get_settings_path())
        self.config = cfg
        self.settings_changed.emit(cfg)
        self.accept()
