"""
ui/training_window.py
Окно запуска обучения персонального LoRA-адаптера.

Секции:
  1. «Данные»   — экспорт Telegram, User ID, кнопка «Обработать»
  2. «Обучение» — гиперпараметры, лог, прогресс, Старт/Стоп
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import AppConfig
from training.data_processor import ProcessResult, TelegramDataProcessor
from training.trainer import LoRATrainer, TrainingConfig

# ---------------------------------------------------------------------------
# Палитра (milky-white + mint/teal)
# ---------------------------------------------------------------------------
_BG = "#F4FAF8"
_BTN = "#FFFFFF"
_ACCENT = "#3BBFB0"
_TEXT = "#1D6B61"
_HOVER = "#DDF3EF"
_DARK = "#E8F6F3"
_BORDER = "#A8DED8"

_SS_WINDOW = f"background:{_BG}; color:{_TEXT};"

_SS_GROUP = (
    f"QGroupBox{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:6px;"
    f"margin-top:10px; font-weight:bold;}}"
    f"QGroupBox::title{{subcontrol-origin:margin;"
    f"subcontrol-position:top left; padding:0 6px;}}"
)

_SS_INPUT = (
    f"QLineEdit{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; padding:4px;}}"
    f"QSpinBox{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; padding:2px;}}"
)

_SS_BTN = (
    f"QPushButton{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px; padding:5px 12px;}}"
    f"QPushButton:hover{{background:{_HOVER};}}"
    f"QPushButton:pressed{{background:{_ACCENT}; color:white;}}"
    f"QPushButton:disabled{{background:{_DARK}; color:{_BORDER};}}"
)

_SS_BTN_ACCENT = (
    f"QPushButton{{background:{_ACCENT}; color:white;"
    f"border:1px solid {_ACCENT}; border-radius:4px;"
    f"padding:6px 16px; font-weight:bold;}}"
    f"QPushButton:hover{{background:#2EA898;}}"
    f"QPushButton:pressed{{background:#267D73;}}"
    f"QPushButton:disabled{{background:{_DARK}; color:{_BORDER};}}"
)

_SS_LOG = (
    f"QTextEdit{{background:{_BTN}; color:{_TEXT};"
    f"border:1px solid {_BORDER}; border-radius:4px;"
    f"font-family:Consolas,monospace; font-size:11px;}}"
)

_SS_PROGRESS = (
    f"QProgressBar{{background:{_DARK}; border:1px solid {_BORDER};"
    f"border-radius:4px; height:14px; text-align:center; color:{_TEXT};}}"
    f"QProgressBar::chunk{{background:{_ACCENT}; border-radius:3px;}}"
)


# ---------------------------------------------------------------------------
# Воркеры (QThread)
# ---------------------------------------------------------------------------

class _DataWorker(QThread):
    """Обрабатывает экспорт Telegram в фоновом потоке."""

    done_signal = pyqtSignal(object)    # ProcessResult
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        export_folder: str,
        user_id: str,
        output_path: str,
    ) -> None:
        super().__init__()
        self._folder = export_folder
        self._user_id = user_id
        self._output = output_path

    def run(self) -> None:
        try:
            processor = TelegramDataProcessor()
            result = processor.process(
                export_folder=self._folder,
                user_id=self._user_id,
                output_path=self._output,
            )
            if result.success:
                train_p, val_p = processor.split_train_val(result.output_file)
                result.output_file = train_p  # передаём train-путь дальше
                self.done_signal.emit(result)
            else:
                self.error_signal.emit(result.error or "Неизвестная ошибка")
        except Exception as exc:  # pylint: disable=broad-except
            self.error_signal.emit(str(exc))


class _TrainerWorker(QThread):
    """Запускает LoRATrainer.train() в фоновом потоке."""

    progress_signal = pyqtSignal(str)      # строка лога
    step_signal = pyqtSignal(int, int)     # (current, total)
    done_signal = pyqtSignal(str)          # путь к адаптеру
    error_signal = pyqtSignal(str)

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self._config = config
        self._trainer: Optional[LoRATrainer] = None

    def run(self) -> None:
        self._trainer = LoRATrainer(
            config=self._config,
            progress_callback=self.progress_signal.emit,
        )
        try:
            self._trainer.train()
            self.done_signal.emit(self._config.output_dir)
        except Exception as exc:  # pylint: disable=broad-except
            self.error_signal.emit(str(exc))

    def request_stop(self) -> None:
        if self._trainer:
            self._trainer.stop()


# ---------------------------------------------------------------------------
# Главное окно обучения
# ---------------------------------------------------------------------------

class TrainingWindow(QWidget):
    """PyQt6-окно для запуска обучения персонального адаптера."""

    def __init__(
        self,
        app_config: AppConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.app_config = app_config

        self._data_result: Optional[ProcessResult] = None
        self._trainer_worker: Optional[_TrainerWorker] = None
        self._data_worker: Optional[_DataWorker] = None

        self.setWindowTitle("LLM-Keyboard — Обучение адаптера")
        self.setMinimumWidth(540)
        self.setStyleSheet(_SS_WINDOW)

        self._build_ui()

    # ------------------------------------------------------------------
    # Построение UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{background:{_BG}; border:none;}}")

        content = QWidget()
        content.setStyleSheet(f"background:{_BG};")
        vbox = QVBoxLayout(content)
        vbox.setSpacing(10)
        vbox.setContentsMargins(4, 4, 4, 4)

        vbox.addWidget(self._build_data_group())
        vbox.addWidget(self._build_train_group())
        vbox.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        # Нижняя строка: Старт / Стоп
        bottom = QHBoxLayout()
        self._btn_start = QPushButton("▶  Запустить обучение")
        self._btn_start.setStyleSheet(_SS_BTN_ACCENT)
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._start_training)

        self._btn_stop = QPushButton("⏹  Остановить")
        self._btn_stop.setStyleSheet(_SS_BTN)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_training)

        bottom.addWidget(self._btn_start, stretch=2)
        bottom.addWidget(self._btn_stop, stretch=1)
        root.addLayout(bottom)

        # Прогресс-бар
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate по умолчанию
        self._progress.setValue(0)
        self._progress.setStyleSheet(_SS_PROGRESS)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # Лог
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(160)
        self._log.setStyleSheet(_SS_LOG)
        root.addWidget(self._log, stretch=1)

        # TODO: Scheduled training
        # self.schedule_checkbox = QCheckBox("Обучать по расписанию")
        # self.schedule_time_from = QTimeEdit()  # с 02:00
        # self.schedule_time_to = QTimeEdit()    # до 06:00

    def _build_data_group(self) -> QGroupBox:
        box = QGroupBox("📂  Данные (экспорт Telegram)")
        box.setStyleSheet(_SS_GROUP)
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        # Папка экспорта
        row1 = QHBoxLayout()
        self._export_edit = QLineEdit()
        self._export_edit.setPlaceholderText("Папка с messages.html ...")
        self._export_edit.setStyleSheet(_SS_INPUT)
        btn_browse_export = QPushButton("Обзор...")
        btn_browse_export.setStyleSheet(_SS_BTN)
        btn_browse_export.clicked.connect(self._browse_export)
        row1.addWidget(self._export_edit, stretch=1)
        row1.addWidget(btn_browse_export)
        layout.addLayout(row1)

        # User ID
        row2 = QHBoxLayout()
        lbl_uid = QLabel("Telegram User ID:")
        lbl_uid.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._uid_edit = QLineEdit()
        self._uid_edit.setPlaceholderText("123456789")
        self._uid_edit.setStyleSheet(_SS_INPUT)
        row2.addWidget(lbl_uid)
        row2.addWidget(self._uid_edit, stretch=1)
        layout.addLayout(row2)

        # Кнопка обработки + статус
        row3 = QHBoxLayout()
        btn_process = QPushButton("⚙  Обработать данные")
        btn_process.setStyleSheet(_SS_BTN)
        btn_process.clicked.connect(self._process_data)
        self._data_status = QLabel("Данные не обработаны")
        self._data_status.setStyleSheet(f"color:{_BORDER};")
        row3.addWidget(btn_process)
        row3.addWidget(self._data_status, stretch=1)
        layout.addLayout(row3)

        return box

    def _build_train_group(self) -> QGroupBox:
        box = QGroupBox("🏋  Настройки обучения")
        box.setStyleSheet(_SS_GROUP)
        form = QFormLayout(box)
        form.setSpacing(6)
        form.setContentsMargins(10, 16, 10, 10)

        # Папка сохранения адаптера
        row_out = QHBoxLayout()
        default_out = str(
            Path(self.app_config.training_output_path).resolve()
        )
        self._out_edit = QLineEdit(default_out)
        self._out_edit.setStyleSheet(_SS_INPUT)
        btn_browse_out = QPushButton("Обзор...")
        btn_browse_out.setStyleSheet(_SS_BTN)
        btn_browse_out.clicked.connect(self._browse_output)
        row_out.addWidget(self._out_edit, stretch=1)
        row_out.addWidget(btn_browse_out)
        form.addRow("Сохранить адаптер:", row_out)

        # CPT адаптер ID (берём из app_config)
        self._cpt_edit = QLineEdit(self.app_config.cpt_adapter_id or "")
        self._cpt_edit.setStyleSheet(_SS_INPUT)
        self._cpt_edit.setPlaceholderText(
            "qzeaq/Qwen3.5-0.8B-telegram-qlora (пусто = без CPT)"
        )
        form.addRow("CPT Adapter ID:", self._cpt_edit)

        self._cpt_rev_edit = QLineEdit(self.app_config.cpt_revision or "")
        self._cpt_rev_edit.setStyleSheet(_SS_INPUT)
        self._cpt_rev_edit.setPlaceholderText("2.8kk-samples-cpt")
        form.addRow("CPT Revision:", self._cpt_rev_edit)

        # Эпохи
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 20)
        self._epochs_spin.setValue(1)
        self._epochs_spin.setStyleSheet(_SS_INPUT)
        form.addRow("Эпохи:", self._epochs_spin)

        # Learning rate
        self._lr_edit = QLineEdit("5e-5")
        self._lr_edit.setStyleSheet(_SS_INPUT)
        form.addRow("Learning Rate:", self._lr_edit)

        # Инфо о batch
        lbl_batch = QLabel(
            "Batch: 1 × grad_accum 8 = effective 8  "
            "(оптимизатор: paged_adamw_8bit, модель: 8-bit)"
        )
        lbl_batch.setStyleSheet(f"color:{_BORDER}; font-size:11px;")
        form.addRow("", lbl_batch)

        return box

    # ------------------------------------------------------------------
    # Обработчики кнопок
    # ------------------------------------------------------------------

    def _browse_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку экспорта Telegram"
        )
        if folder:
            self._export_edit.setText(folder)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Папка для сохранения адаптера"
        )
        if folder:
            self._out_edit.setText(folder)

    def _process_data(self) -> None:
        folder = self._export_edit.text().strip()
        uid = self._uid_edit.text().strip()

        if not folder or not uid:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Укажите папку экспорта и Telegram User ID.",
            )
            return

        out_dir = Path(self.app_config.training_output_path).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = str(out_dir / "personal.jsonl")

        self._data_status.setText("⏳ Обработка...")
        self._data_status.setStyleSheet(f"color:{_ACCENT};")
        self._log_append("⏳ Начало обработки экспорта Telegram...")

        self._data_worker = _DataWorker(folder, uid, jsonl_path)
        self._data_worker.done_signal.connect(self._on_data_done)
        self._data_worker.error_signal.connect(self._on_data_error)
        self._data_worker.start()

    def _on_data_done(self, result: ProcessResult) -> None:
        self._data_result = result
        tokens = result.total_tokens_estimate
        msgs = result.total_messages
        chunks = result.total_chunks

        status_text = (
            f"✅ {msgs} сообщений → {chunks} чанков "
            f"(~{tokens:,} токенов)"
        )
        self._data_status.setText(status_text)
        self._data_status.setStyleSheet(f"color:{_ACCENT};")
        self._log_append(status_text)

        if result.has_enough_tokens:
            self._btn_start.setEnabled(True)
        else:
            warn = (
                f"⚠️  Токенов мало (~{tokens:,}). "
                f"Рекомендуется ≥10 000. "
                f"Обучение возможно, но качество может быть низким."
            )
            self._log_append(warn)
            self._data_status.setText(
                f"⚠️  ~{tokens:,} токенов (мало!)"
            )
            self._data_status.setStyleSheet("color:#CC6633;")
            self._btn_start.setEnabled(True)  # разрешаем, но предупреждаем

    def _on_data_error(self, msg: str) -> None:
        self._data_status.setText("❌ Ошибка")
        self._data_status.setStyleSheet("color:#CC3333;")
        self._log_append(f"❌ Ошибка обработки: {msg}")

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        if self._data_result is None:
            QMessageBox.warning(self, "Ошибка", "Сначала обработайте данные.")
            return

        data_path = Path(self._data_result.output_file)
        val_path = data_path.parent / "val.jsonl"
        if not data_path.exists() or not val_path.exists():
            QMessageBox.warning(
                self,
                "Ошибка",
                "Файлы train.jsonl / val.jsonl не найдены.\n"
                "Повторите обработку данных.",
            )
            return

        try:
            lr = float(self._lr_edit.text().strip())
        except ValueError:
            QMessageBox.warning(
                self, "Ошибка", "Некорректный learning rate."
            )
            return

        cpt_id = self._cpt_edit.text().strip() or None
        cpt_rev = self._cpt_rev_edit.text().strip() or None

        cfg = TrainingConfig(
            train_path=str(data_path),
            val_path=str(val_path),
            output_dir=self._out_edit.text().strip(),
            model_id="unsloth/Qwen3.5-0.8B-Base",
            cpt_adapter_id=cpt_id,
            cpt_revision=cpt_rev,
            num_epochs=self._epochs_spin.value(),
            learning_rate=lr,
            batch_size=1,
            grad_accum=8,
        )

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate

        self._trainer_worker = _TrainerWorker(cfg)
        self._trainer_worker.progress_signal.connect(self._log_append)
        self._trainer_worker.done_signal.connect(self._on_train_done)
        self._trainer_worker.error_signal.connect(self._on_train_error)
        self._trainer_worker.start()

    def _stop_training(self) -> None:
        if self._trainer_worker:
            self._trainer_worker.request_stop()
        self._btn_stop.setEnabled(False)

    def _on_train_done(self, output_dir: str) -> None:
        self._progress.setVisible(False)
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._log_append(
            f"✅ Обучение завершено. Адаптер сохранён: {output_dir}"
        )
        QMessageBox.information(
            self,
            "Готово",
            f"Обучение завершено!\nАдаптер: {output_dir}",
        )

    def _on_train_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._btn_start.setEnabled(bool(self._data_result))
        self._btn_stop.setEnabled(False)
        self._log_append(f"❌ Ошибка обучения: {msg}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка обучения:\n{msg}")

    # ------------------------------------------------------------------
    # Лог
    # ------------------------------------------------------------------

    def _log_append(self, text: str) -> None:
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )
