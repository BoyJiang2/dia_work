from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from image_io import cv_to_qpixmap, imread_unicode, imwrite_unicode, to_gray
from operations import OPERATIONS, Operation, ParamSpec, by_category, categories


class Card(QFrame):
    def __init__(self, title: str | None = None) -> None:
        super().__init__()
        self.setObjectName("card")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(31, 50, 72, 34))
        self.setGraphicsEffect(shadow)

        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 18)
        self.body.setSpacing(12)
        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            self.body.addWidget(label)


class ImageCanvas(QLabel):
    def __init__(self, empty_text: str) -> None:
        super().__init__()
        self.empty_text = empty_text
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("imageCanvas")
        self.setText(empty_text)

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = None
            self.clear()
            self.setText(self.empty_text)
            return
        self._pixmap = cv_to_qpixmap(image)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class ImagePanel(QFrame):
    def __init__(self, title: str, badge: str, empty_text: str) -> None:
        super().__init__()
        self.setObjectName("imagePanel")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("imageTitle")
        self.badge_label = QLabel(badge)
        self.badge_label.setObjectName("imageBadge")
        self.meta_label = QLabel("未载入")
        self.meta_label.setObjectName("imageMeta")
        self.canvas = ImageCanvas(empty_text)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        header.addWidget(self.title_label)
        header.addWidget(self.badge_label)
        header.addStretch(1)
        header.addWidget(self.meta_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self.canvas, stretch=1)

    def set_image(self, image: np.ndarray | None) -> None:
        self.canvas.set_image(image)
        if image is None:
            self.meta_label.setText("未载入")
            return
        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        self.meta_label.setText(f"{w} x {h} / {channels} 通道")


class MetricChip(QFrame):
    def __init__(self, name: str, value: str = "-") -> None:
        super().__init__()
        self.setObjectName("metricChip")
        self.name_label = QLabel(name)
        self.name_label.setObjectName("metricName")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("rootWindow")
        self.setWindowTitle("数字图像处理综合实验系统")
        self.resize(1440, 880)
        self.original: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.current_path: Path | None = None
        self.current_operation: Operation = OPERATIONS[0]
        self.param_widgets: dict[str, QWidget] = {}
        self._building_params = False

        self.category_box = QComboBox()
        self.operation_box = QComboBox()
        self.description_label = QLabel()
        self.description_label.setObjectName("description")
        self.description_label.setWordWrap(True)
        self.params_layout = QFormLayout()
        self.params_layout.setLabelAlignment(Qt.AlignLeft)
        self.params_layout.setFormAlignment(Qt.AlignTop)
        self.params_layout.setHorizontalSpacing(12)
        self.params_layout.setVerticalSpacing(12)
        self.history = QListWidget()
        self.history.setObjectName("historyList")

        self.original_view = ImagePanel("原始图像", "INPUT", "打开图像后在这里显示原图")
        self.result_view = ImagePanel("处理结果", "OUTPUT", "执行算法后在这里显示结果")
        self.mean_chip = MetricChip("平均灰度")
        self.std_chip = MetricChip("标准差")
        self.mse_chip = MetricChip("MSE")
        self.psnr_chip = MetricChip("PSNR")
        self.ssim_chip = MetricChip("SSIM")
        self.path_label = QLabel("未选择文件")
        self.path_label.setObjectName("pathLabel")
        self.path_label.setWordWrap(True)

        self._build_ui()
        self._connect()
        self._load_categories()
        self._update_metrics()

    def _build_ui(self) -> None:
        self.open_btn = QPushButton("打开图像")
        self.open_btn.setObjectName("primaryButton")
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setObjectName("secondaryButton")
        self.apply_btn = QPushButton("执行处理")
        self.apply_btn.setObjectName("accentButton")
        self.reset_btn = QPushButton("结果作为原图")
        self.reset_btn.setObjectName("ghostButton")
        self.auto_apply_box = QCheckBox("参数变化时自动预览")
        self.auto_apply_box.setChecked(True)

        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(22, 18, 22, 18)
        shell_layout.setSpacing(16)
        shell_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_workspace())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1050])
        shell_layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(shell)
        self.setStatusBar(QStatusBar())
        self.statusBar().setObjectName("status")
        self.statusBar().showMessage("就绪")
        self._set_style()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("topBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(18)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(4)
        title = QLabel("数字图像处理综合实验系统")
        title.setObjectName("appTitle")
        subtitle = QLabel("算法验证 / 参数实验 / 结果对比 / 质量评价")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        course_tag = QLabel("CV LAB")
        course_tag.setObjectName("courseTag")
        layout.addLayout(title_box, stretch=1)
        layout.addWidget(course_tag)
        return header

    def _build_control_panel(self) -> QWidget:
        container = QWidget()
        container.setObjectName("controlContainer")
        container.setMinimumWidth(370)
        container.setMaximumWidth(440)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        quick_card = Card("工作流")
        quick_card.body.addWidget(self.open_btn)
        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        quick_row.addWidget(self.save_btn)
        quick_row.addWidget(self.reset_btn)
        quick_card.body.addLayout(quick_row)
        quick_card.body.addWidget(self.path_label)
        layout.addWidget(quick_card)

        op_card = Card("算法选择")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.addRow("类别", self.category_box)
        form.addRow("方法", self.operation_box)
        op_card.body.addLayout(form)
        op_card.body.addWidget(self.description_label)
        layout.addWidget(op_card)

        param_card = Card("参数调节")
        param_card.body.addLayout(self.params_layout)
        param_card.body.addWidget(self.auto_apply_box)
        param_card.body.addWidget(self.apply_btn)
        layout.addWidget(param_card)

        metrics_card = Card("质量指标")
        for row_widgets in ((self.mean_chip, self.std_chip), (self.mse_chip, self.psnr_chip), (self.ssim_chip,)):
            row = QHBoxLayout()
            row.setSpacing(10)
            for widget in row_widgets:
                row.addWidget(widget)
            metrics_card.body.addLayout(row)
        layout.addWidget(metrics_card)

        history_card = Card("执行历史")
        history_card.body.addWidget(self.history)
        layout.addWidget(history_card, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(460)
        return scroll

    def _build_workspace(self) -> QWidget:
        workspace = QFrame()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        stage_header = QHBoxLayout()
        stage_header.setContentsMargins(2, 0, 2, 0)
        title = QLabel("图像处理工作台")
        title.setObjectName("stageTitle")
        hint = QLabel("左侧选择方法并调节参数，右侧对比处理效果")
        hint.setObjectName("stageHint")
        stage_header.addWidget(title)
        stage_header.addStretch(1)
        stage_header.addWidget(hint)
        layout.addLayout(stage_header)

        image_row = QHBoxLayout()
        image_row.setSpacing(14)
        image_row.addWidget(self.original_view, stretch=1)
        image_row.addWidget(self.result_view, stretch=1)
        layout.addLayout(image_row, stretch=1)
        return workspace

    def _set_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
                font-size: 14px;
                color: #18202f;
            }
            QMainWindow#rootWindow, QWidget#appShell {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #edf3ff, stop:0.45 #f7fbff, stop:1 #eef7f4);
            }
            QFrame#topBar {
                border: 1px solid rgba(255,255,255,0.72);
                border-radius: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f3d68, stop:0.52 #1c6aa6, stop:1 #2aa889);
            }
            QLabel#appTitle {
                color: white;
                font-size: 25px;
                font-weight: 700;
                letter-spacing: 0px;
            }
            QLabel#appSubtitle {
                color: rgba(255,255,255,0.78);
                font-size: 13px;
            }
            QLabel#courseTag {
                color: white;
                font-weight: 700;
                padding: 8px 14px;
                border: 1px solid rgba(255,255,255,0.38);
                border-radius: 8px;
                background: rgba(255,255,255,0.14);
            }
            QFrame#card {
                border: 1px solid rgba(201, 212, 226, 0.72);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.92);
            }
            QLabel#cardTitle {
                color: #1d2b3f;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#description, QLabel#pathLabel {
                color: #65748a;
                line-height: 1.4;
                padding: 8px 10px;
                border-radius: 8px;
                background: #f3f7fb;
            }
            QPushButton {
                min-height: 36px;
                border: 0;
                border-radius: 8px;
                padding: 8px 13px;
                font-weight: 650;
            }
            QPushButton#primaryButton {
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1769e0, stop:1 #20a4a6);
            }
            QPushButton#accentButton {
                color: white;
                min-height: 42px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f06a3d, stop:1 #db2f6f);
            }
            QPushButton#secondaryButton {
                color: #17567f;
                background: #e7f2fb;
                border: 1px solid #bdd9ed;
            }
            QPushButton#ghostButton {
                color: #3a4b61;
                background: #f0f4f8;
                border: 1px solid #d8e0e8;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 32px;
                border: 1px solid #c8d4e2;
                border-radius: 8px;
                padding: 4px 9px;
                background: white;
                selection-background-color: #1c6aa6;
            }
            QComboBox::drop-down {
                width: 26px;
                border: 0;
            }
            QCheckBox {
                spacing: 8px;
                color: #304158;
            }
            QScrollArea {
                background: transparent;
            }
            QFrame#workspace {
                border-radius: 12px;
                border: 1px solid rgba(37, 54, 76, 0.18);
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #172130, stop:0.5 #1d2f44, stop:1 #173d41);
            }
            QLabel#stageTitle {
                color: #f4f8ff;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#stageHint {
                color: #9fb1c8;
                font-size: 13px;
            }
            QFrame#imagePanel {
                border: 1px solid rgba(197, 214, 232, 0.18);
                border-radius: 10px;
                background: rgba(10, 16, 25, 0.34);
            }
            QLabel#imageTitle {
                color: #f2f7ff;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#imageBadge {
                color: #7ee4cf;
                font-size: 11px;
                font-weight: 800;
                padding: 3px 7px;
                border-radius: 6px;
                background: rgba(126, 228, 207, 0.12);
                border: 1px solid rgba(126, 228, 207, 0.24);
            }
            QLabel#imageMeta {
                color: #9eb0c5;
                font-size: 12px;
            }
            QLabel#imageCanvas {
                color: #aebed2;
                border: 1px dashed rgba(178, 197, 218, 0.24);
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #101722, stop:0.55 #162233, stop:1 #0e1e23);
            }
            QFrame#metricChip {
                border: 1px solid #dbe4ee;
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffffff, stop:1 #f0f7fb);
            }
            QLabel#metricName {
                color: #748398;
                font-size: 12px;
            }
            QLabel#metricValue {
                color: #14233a;
                font-size: 16px;
                font-weight: 750;
            }
            QListWidget#historyList {
                min-height: 110px;
                border: 1px solid #d7e0ea;
                border-radius: 8px;
                background: #fbfdff;
                padding: 6px;
            }
            QListWidget#historyList::item {
                padding: 8px;
                border-radius: 6px;
                color: #33445a;
            }
            QListWidget#historyList::item:selected {
                color: white;
                background: #1c6aa6;
            }
            QSplitter::handle {
                background: transparent;
                width: 10px;
            }
            QStatusBar#status {
                color: #526176;
                background: transparent;
            }
            """
        )

    def _connect(self) -> None:
        self.open_btn.clicked.connect(self.open_image)
        self.save_btn.clicked.connect(self.save_result)
        self.apply_btn.clicked.connect(self.apply_operation)
        self.reset_btn.clicked.connect(self.promote_result)
        self.category_box.currentTextChanged.connect(self._load_operations)
        self.operation_box.currentIndexChanged.connect(self._operation_changed)

    def _load_categories(self) -> None:
        self.category_box.clear()
        self.category_box.addItems(categories())

    def _load_operations(self, category: str) -> None:
        self.operation_box.clear()
        for op in by_category(category):
            self.operation_box.addItem(op.name, op.key)
        self._operation_changed(0)

    def _operation_changed(self, index: int) -> None:
        if index < 0:
            return
        key = self.operation_box.itemData(index)
        self.current_operation = next(op for op in OPERATIONS if op.key == key)
        self.description_label.setText(self.current_operation.description)
        self._build_params(self.current_operation)
        self._auto_apply()

    def _build_params(self, operation: Operation) -> None:
        self._building_params = True
        while self.params_layout.rowCount():
            self.params_layout.removeRow(0)
        self.param_widgets.clear()
        if not operation.params:
            empty = QLabel("该方法无需额外参数")
            empty.setObjectName("description")
            self.params_layout.addRow(empty)
            self._building_params = False
            return
        for spec in operation.params:
            widget = self._create_param_widget(spec)
            self.param_widgets[spec.name] = widget
            self.params_layout.addRow(spec.label, widget)
        self._building_params = False

    def _create_param_widget(self, spec: ParamSpec) -> QWidget:
        if spec.kind == "int":
            widget = QSpinBox()
            widget.setRange(int(spec.minimum or 0), int(spec.maximum or 9999))
            widget.setSingleStep(int(spec.step))
            widget.setValue(int(spec.default))
            widget.valueChanged.connect(self._auto_apply)
            return widget
        if spec.kind == "float":
            widget = QDoubleSpinBox()
            minimum = float(spec.minimum if spec.minimum is not None else -9999)
            maximum = float(spec.maximum if spec.maximum is not None else 9999)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(float(spec.step))
            widget.setDecimals(3)
            widget.setValue(float(spec.default))
            widget.valueChanged.connect(self._auto_apply)
            return widget
        if spec.kind == "choice":
            widget = QComboBox()
            widget.addItems(spec.choices)
            widget.setCurrentText(str(spec.default))
            widget.currentTextChanged.connect(self._auto_apply)
            return widget
        widget = QCheckBox()
        widget.setChecked(bool(spec.default))
        widget.stateChanged.connect(self._auto_apply)
        return widget

    def _params(self) -> dict:
        values = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
        return values

    def _auto_apply(self) -> None:
        if self._building_params or not self.auto_apply_box.isChecked() or self.original is None:
            return
        self.apply_operation(add_history=False)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        try:
            self.original = imread_unicode(path)
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self.result = None
        self.current_path = Path(path)
        self.original_view.set_image(self.original)
        self.result_view.set_image(None)
        self.history.clear()
        self.path_label.setText(str(path))
        self.statusBar().showMessage(f"已载入 {path}")
        self._update_metrics()
        self._auto_apply()

    def save_result(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "提示", "请先执行图像处理。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            str(Path.cwd() / "result.png"),
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;TIFF (*.tif);;All Files (*)",
        )
        if not path:
            return
        try:
            imwrite_unicode(path, self.result)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.statusBar().showMessage(f"已保存 {path}")

    def apply_operation(self, add_history: bool = True) -> None:
        if self.original is None:
            QMessageBox.information(self, "提示", "请先打开一张图像。")
            return
        try:
            params = self._params()
            self.result = self.current_operation.apply(self.original.copy(), params)
        except Exception as exc:
            QMessageBox.critical(self, "处理失败", str(exc))
            return
        self.result_view.set_image(self.result)
        if add_history:
            self.history.insertItem(0, f"{self.current_operation.category} / {self.current_operation.name}  {params}")
        self._update_metrics()
        self.statusBar().showMessage("处理完成")

    def promote_result(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "提示", "当前没有处理结果。")
            return
        self.original = self.result.copy()
        self.result = None
        self.original_view.set_image(self.original)
        self.result_view.set_image(None)
        self._update_metrics()
        self.statusBar().showMessage("已将结果设为新的原图")

    def _update_metrics(self) -> None:
        if self.original is None:
            for chip in (self.mean_chip, self.std_chip, self.mse_chip, self.psnr_chip, self.ssim_chip):
                chip.set_value("-")
            return
        gray = to_gray(self.result if self.result is not None else self.original)
        self.mean_chip.set_value(f"{gray.mean():.2f}")
        self.std_chip.set_value(f"{gray.std():.2f}")
        if self.result is None:
            self.mse_chip.set_value("-")
            self.psnr_chip.set_value("-")
            self.ssim_chip.set_value("-")
            return
        mse, psnr, ssim = self._diff_metrics(self.original, self.result)
        self.mse_chip.set_value(f"{mse:.2f}")
        self.psnr_chip.set_value(psnr)
        self.ssim_chip.set_value(f"{ssim:.4f}")

    def _diff_metrics(self, a: np.ndarray, b: np.ndarray) -> tuple[float, str, float]:
        bg = to_gray(b)
        ag = cv2.resize(to_gray(a), (bg.shape[1], bg.shape[0]))
        af = ag.astype(np.float32)
        bf = bg.astype(np.float32)
        mse = float(np.mean((af - bf) ** 2))
        psnr = "inf" if mse == 0 else f"{20 * np.log10(255.0 / np.sqrt(mse)):.2f} dB"

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        mu_a = float(af.mean())
        mu_b = float(bf.mean())
        var_a = float(af.var())
        var_b = float(bf.var())
        cov = float(((af - mu_a) * (bf - mu_b)).mean())
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
        return mse, psnr, float(np.clip(ssim, -1.0, 1.0))


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
