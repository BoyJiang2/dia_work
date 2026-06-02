from __future__ import annotations

import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QShortcut,
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from image_io import cv_to_qpixmap, imread_unicode, imwrite_unicode, to_gray
from operations import OPERATIONS, Operation, ParamSpec, by_category, categories
from themes import GLOBAL_QSS


class Card(QFrame):
    def __init__(self, title: str | None = None) -> None:
        super().__init__()
        self.setObjectName("card")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 18)
        self.body.setSpacing(12)
        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            self.body.addWidget(label)


class ImageCanvas(QLabel):
    pixel_hovered = pyqtSignal(int, int, object)  # x, y, pixel_value_or_None

    def __init__(self, empty_text: str) -> None:
        super().__init__()
        self.empty_text = empty_text
        self._pixmap: QPixmap | None = None
        self._image: np.ndarray | None = None  # 保留原始数组用于像素查询
        self._zoom: float = 1.0  # 缩放因子，1.0 = 适合窗口
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("imageCanvas")
        self.setText(empty_text)
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = None
            self._image = None
            self._zoom = 1.0
            self.clear()
            self.setText(self.empty_text)
            return
        self._image = image
        self._pixmap = cv_to_qpixmap(image)
        self._zoom = 1.0
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap
        if abs(self._zoom - 1.0) < 0.001:
            # 适应窗口
            scaled = scaled.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            w = int(self._pixmap.width() * self._zoom)
            h = int(self._pixmap.height() * self._zoom)
            scaled = scaled.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def _image_coords(self, pos) -> tuple[int, int] | None:
        """将控件坐标映射到图像像素坐标."""
        if self._pixmap is None or self._image is None:
            return None
        pix = self.pixmap()
        if pix is None:
            return None
        # 计算显示的偏移
        ox = (self.width() - pix.width()) // 2
        oy = (self.height() - pix.height()) // 2
        ix = pos.x() - ox
        iy = pos.y() - oy
        if ix < 0 or iy < 0 or ix >= pix.width() or iy >= pix.height():
            return None
        # 映射回原始图像坐标
        img_h, img_w = self._image.shape[:2]
        px = int(ix * img_w / pix.width())
        py = int(iy * img_h / pix.height())
        return px, py

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        coords = self._image_coords(event.pos())
        if coords is None:
            self.pixel_hovered.emit(-1, -1, None)
            return
        px, py = coords
        if self._image is not None and 0 <= py < self._image.shape[0] and 0 <= px < self._image.shape[1]:
            val = self._image[py, px]
            self.pixel_hovered.emit(px, py, val)
        else:
            self.pixel_hovered.emit(-1, -1, None)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.pixel_hovered.emit(-1, -1, None)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom = min(10.0, self._zoom * 1.15)
            else:
                self._zoom = max(0.05, self._zoom / 1.15)
            # 接近 1.0 时吸附
            if abs(self._zoom - 1.0) < 0.02:
                self._zoom = 1.0
            self._rescale()
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        if abs(self._zoom - 1.0) < 0.01:
            # 当前适应窗口 → 切换到 100% (1:1)
            self._zoom = 1.0
            self._rescale()
            # 计算 100% 需要的缩放（按原始图像尺寸 vs 显示尺寸）
            # 实际上 100% = 原始像素 1:1 映射
            img_w = self._pixmap.width()
            view_w = self.width()
            if img_w > view_w:
                self._zoom = 1.0  # 保持原图大小（pixmap 本身就是原图大小）
            else:
                self._zoom = 1.0
            self._rescale()
        else:
            # 从缩放状态 → 适应窗口
            self._zoom = 1.0
            self._rescale()


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
        self._history_data: list[np.ndarray] = []  # 与 history 条目对应的结果图

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
        self.open_btn = QPushButton("📂  打开图像")
        self.open_btn.setObjectName("primaryButton")
        self.save_btn = QPushButton("💾  保存结果")
        self.save_btn.setObjectName("secondaryButton")
        self.apply_btn = QPushButton("▶  执行处理")
        self.apply_btn.setObjectName("accentButton")
        self.reset_btn = QPushButton("↻  结果作为原图")
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
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(18)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # 不确定模式
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("就绪")
        self.setStyleSheet(GLOBAL_QSS)

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

        course_tag = QLabel("◆  CV  LAB")
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
        self.batch_btn = QPushButton("📦  批处理")
        self.batch_btn.setObjectName("ghostButton")
        quick_card.body.addWidget(self.batch_btn)
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
        self.compare_box = QCheckBox("叠加对比")
        self.compare_box.setObjectName("compareCheck")
        self.compare_box.setToolTip("在结果图上半透明叠加原图，便于对比差异")
        self.profile_btn = QPushButton("📈 剖线图")
        self.profile_btn.setObjectName("toolBtn")
        self.profile_btn.setToolTip("查看图像水平/垂直剖线强度图")
        self.surface_btn = QPushButton("⛰ 3D视图")
        self.surface_btn.setObjectName("toolBtn")
        self.surface_btn.setToolTip("查看图像像素强度的3D曲面图")
        self.report_btn = QPushButton("📄 导出报告")
        self.report_btn.setObjectName("toolBtn")
        self.report_btn.setToolTip("导出HTML实验报告（原图/结果/参数/指标）")
        stage_header.addWidget(title)
        stage_header.addStretch(1)
        stage_header.addWidget(self.profile_btn)
        stage_header.addWidget(self.surface_btn)
        stage_header.addWidget(self.report_btn)
        stage_header.addWidget(self.compare_box)
        stage_header.addWidget(hint)
        layout.addLayout(stage_header)

        image_row = QHBoxLayout()
        image_row.setSpacing(14)
        image_row.addWidget(self.original_view, stretch=1)
        image_row.addWidget(self.result_view, stretch=1)
        layout.addLayout(image_row, stretch=1)
        return workspace

    def _connect(self) -> None:
        self.open_btn.clicked.connect(self.open_image)
        self.save_btn.clicked.connect(self.save_result)
        self.apply_btn.clicked.connect(self.apply_operation)
        self.reset_btn.clicked.connect(self.promote_result)
        self.batch_btn.clicked.connect(self._batch_process)
        self.profile_btn.clicked.connect(self._show_line_profile)
        self.surface_btn.clicked.connect(self._show_3d_surface)
        self.report_btn.clicked.connect(self._export_report)
        self.category_box.currentTextChanged.connect(self._load_operations)
        self.operation_box.currentIndexChanged.connect(self._operation_changed)
        # 历史记录点击回溯
        self.history.itemClicked.connect(self._on_history_clicked)
        # 叠加对比
        self.compare_box.stateChanged.connect(self._on_compare_toggled)
        # 像素探查
        self.original_view.canvas.pixel_hovered.connect(self._on_original_hover)
        self.result_view.canvas.pixel_hovered.connect(self._on_result_hover)
        # 键盘快捷键
        for key, slot in ((QKeySequence.Open, self.open_image),
                          (QKeySequence.Save, self.save_result),
                          (QKeySequence("Ctrl+Return"), self.apply_operation),
                          (QKeySequence("Ctrl+R"), self.promote_result),
                          (QKeySequence("Ctrl+Shift+S"), self.save_result)):
            sc = QShortcut(key, self)
            sc.activated.connect(slot)

    def _on_history_clicked(self, item) -> None:
        """点击历史记录条目，恢复对应的处理结果."""
        idx = self.history.row(item)
        if 0 <= idx < len(self._history_data):
            self.result = self._history_data[idx].copy()
            if self.compare_box.isChecked():
                self._on_compare_toggled()
            else:
                self.result_view.set_image(self.result)
            self._update_metrics()
            self.statusBar().showMessage(f"已回溯: {item.text()[:60]}")

    def _on_compare_toggled(self) -> None:
        """切换叠加对比模式：在结果图上 50% 叠加原图."""
        if self.compare_box.isChecked() and self.original is not None and self.result is not None:
            # 确保尺寸一致
            orig = self.original
            res = self.result
            if orig.shape[:2] != res.shape[:2]:
                res = cv2.resize(res, (orig.shape[1], orig.shape[0]))
            if orig.ndim != res.ndim:
                if orig.ndim == 2:
                    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                if res.ndim == 2:
                    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(orig, 0.5, res, 0.5, 0)
            self.result_view.set_image(blended)
            self.statusBar().showMessage("叠加对比：原图 50% + 结果 50%")
        else:
            if self.result is not None:
                self.result_view.set_image(self.result)
                self.statusBar().showMessage("对比模式关闭")

    def _on_original_hover(self, x: int, y: int, val) -> None:
        if val is None:
            return
        if isinstance(val, np.ndarray):
            if val.size == 1:
                self.statusBar().showMessage(f"原图 ({x},{y}) = {int(val.flat[0])}")
            else:
                b, g, r = int(val[0]), int(val[1]), int(val[2])
                self.statusBar().showMessage(f"原图 ({x},{y}) BGR=({b},{g},{r})")
        else:
            self.statusBar().showMessage(f"原图 ({x},{y}) = {int(val)}")

    def _on_result_hover(self, x: int, y: int, val) -> None:
        if val is None:
            return
        if isinstance(val, np.ndarray):
            if val.size == 1:
                self.statusBar().showMessage(f"结果 ({x},{y}) = {int(val.flat[0])}")
            else:
                b, g, r = int(val[0]), int(val[1]), int(val[2])
                self.statusBar().showMessage(f"结果 ({x},{y}) BGR=({b},{g},{r})")
        else:
            self.statusBar().showMessage(f"结果 ({x},{y}) = {int(val)}")

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
        self._history_data.clear()
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
        if self.compare_box.isChecked():
            self._on_compare_toggled()
        else:
            self.result_view.set_image(self.result)
        if add_history:
            desc = f"{self.current_operation.category} / {self.current_operation.name}  {params}"
            self.history.insertItem(0, desc)
            self._history_data.insert(0, self.result.copy())
            # 限制历史记录数量
            if len(self._history_data) > 50:
                self._history_data.pop()
                self.history.takeItem(self.history.count() - 1)
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

    # ---- 批处理 ----

    def _batch_process(self) -> None:
        """对文件夹内所有图片执行当前算法并保存结果."""
        if self.original is None:
            QMessageBox.information(self, "提示", "请先打开一张图片以选择操作和参数。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择包含图像的文件夹", str(Path.cwd()))
        if not folder:
            return
        folder = Path(folder)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
        if not images:
            QMessageBox.information(self, "提示", f"文件夹中没有找到图像文件。")
            return

        out_dir = folder / f"batch_{self.current_operation.key}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(images))
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"批处理 0/{len(images)}...")

        op = self.current_operation
        params = self._params()

        for i, img_path in enumerate(images):
            try:
                img = imread_unicode(str(img_path))
                result = op.apply(img.copy(), params)
                out_path = out_dir / f"{img_path.stem}_{op.key}{img_path.suffix}"
                imwrite_unicode(str(out_path), result)
            except Exception as exc:
                QMessageBox.warning(self, "批处理警告", f"处理 {img_path.name} 失败:\n{exc}")
            self.progress_bar.setValue(i + 1)
            self.statusBar().showMessage(f"批处理 {i + 1}/{len(images)}: {img_path.name}")
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage(f"批处理完成，结果保存在 {out_dir}")

    # ---- 剖线图 ----

    def _show_line_profile(self) -> None:
        """显示当前图像（原始或结果）的水平/垂直剖线强度图."""
        target = self.result if self.result is not None else self.original
        if target is None:
            QMessageBox.information(self, "提示", "请先打开图像。")
            return
        image = target
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        cy, cx = h // 2, w // 2

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), num="剖线图")

        # 原图 + 十字线
        ax_img = axes[0, 0]
        ax_img.imshow(gray, cmap="gray")
        ax_img.axhline(cy, color="r", linewidth=1)
        ax_img.axvline(cx, color="r", linewidth=1)
        ax_img.set_title(f"Image {w}x{h} (crosshair at center)")
        ax_img.set_xlabel("X")
        ax_img.set_ylabel("Y")

        # 水平剖线
        ax_h = axes[0, 1]
        ax_h.plot(range(w), gray[cy, :], "b-", linewidth=0.8)
        ax_h.axvline(cx, color="r", linestyle="--", linewidth=0.5)
        ax_h.set_title(f"Horizontal profile at y={cy}")
        ax_h.set_xlabel("X (pixel)")
        ax_h.set_ylabel("Intensity")
        ax_h.set_xlim(0, w)
        ax_h.grid(True, alpha=0.3)

        # 垂直剖线
        ax_v = axes[1, 0]
        ax_v.plot(gray[:, cx], range(h), "g-", linewidth=0.8)
        ax_v.axhline(cy, color="r", linestyle="--", linewidth=0.5)
        ax_v.set_title(f"Vertical profile at x={cx}")
        ax_v.set_xlabel("Intensity")
        ax_v.set_ylabel("Y (pixel)")
        ax_v.set_ylim(h, 0)
        ax_v.grid(True, alpha=0.3)

        # 直方图
        ax_hist = axes[1, 1]
        ax_hist.hist(gray.ravel(), bins=64, color="steelblue", edgecolor="none", alpha=0.85)
        ax_hist.set_title("Histogram")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ---- 3D 曲面图 ----

    def _show_3d_surface(self) -> None:
        """显示当前图像的3D像素强度曲面图."""
        target = self.result if self.result is not None else self.original
        if target is None:
            QMessageBox.information(self, "提示", "请先打开图像。")
            return
        image = target
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 对较大图像降采样
        max_dim = 256
        h, w = gray.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = gray.shape

        fig = plt.figure(figsize=(12, 9), num="3D曲面图")
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(range(w), range(h))
        ax.plot_surface(X, Y, gray, cmap="terrain", linewidth=0, antialiased=True, alpha=0.92)
        ax.set_title(f"3D Intensity Surface ({w}x{h})")
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        ax.set_zlabel("Intensity")
        ax.view_init(elev=35, azim=-60)

        # 添加底部投影
        ax.contourf(X, Y, gray, zdir="z", offset=gray.min() - 30, cmap="terrain", alpha=0.5)

        plt.tight_layout()
        plt.show()

    # ---- 报告导出 ----

    def _export_report(self) -> None:
        """导出 HTML 实验报告."""
        if self.original is None:
            QMessageBox.information(self, "提示", "请先打开图像。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存 HTML 报告", str(Path.cwd() / "report.html"), "HTML (*.html)")
        if not path:
            return

        # 将图像编码为 base64
        import base64

        def img_to_b64(img: np.ndarray) -> str:
            if img.ndim == 2:
                disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                disp = img
            ok, enc = cv2.imencode(".png", disp)
            if ok:
                return base64.b64encode(enc.tobytes()).decode()
            return ""

        orig_b64 = img_to_b64(self.original)
        result_b64 = img_to_b64(self.result) if self.result is not None else ""

        # 收集质量指标
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY) if self.original.ndim == 3 else self.original
        mean_val = f"{gray.mean():.2f}"
        std_val = f"{gray.std():.2f}"
        mse_val, psnr_val, ssim_val = ("-", "-", "-")
        if self.result is not None:
            mse_val, psnr_val, ssim_val = self._diff_metrics(self.original, self.result)
            mse_val = f"{mse_val:.2f}"
            ssim_val = f"{ssim_val:.4f}"

        params = self._params() if self.result is not None else {}
        params_html = "<br>".join(f"{k} = {v}" for k, v in params.items()) or "无"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>数字图像处理实验报告</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 1100px; margin: 30px auto; padding: 20px; background: #f5f7fa; color: #222; }}
  h1 {{ text-align: center; color: #0f3d68; }}
  .meta {{ text-align: center; color: #666; font-size: 13px; margin-bottom: 30px; }}
  .section {{ background: white; border-radius: 10px; padding: 20px; margin: 18px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .section h2 {{ color: #1c6aa6; margin-top: 0; border-bottom: 2px solid #e0e8f0; padding-bottom: 8px; }}
  .images {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .img-card {{ flex: 1; min-width: 300px; text-align: center; }}
  .img-card img {{ max-width: 100%; border-radius: 6px; border: 1px solid #ddd; }}
  .img-card p {{ color: #555; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #dde; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f5fa; color: #1c6aa6; }}
  .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>数字图像处理实验报告</h1>
<div class="meta">
  生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} &nbsp;|&nbsp;
  文件: {self.current_path.name if self.current_path else "无"} &nbsp;|&nbsp;
  算法: {self.current_operation.category} / {self.current_operation.name}
</div>
<div class="section">
  <h2>算法信息</h2>
  <table>
    <tr><th>类别</th><td>{self.current_operation.category}</td></tr>
    <tr><th>方法</th><td>{self.current_operation.name}</td></tr>
    <tr><th>说明</th><td>{self.current_operation.description}</td></tr>
    <tr><th>参数</th><td>{params_html}</td></tr>
  </table>
</div>
<div class="section">
  <h2>图像对比</h2>
  <div class="images">
    <div class="img-card"><p>原始图像</p><img src="data:image/png;base64,{orig_b64}" alt="original"></div>
    {"<div class='img-card'><p>处理结果</p><img src='data:image/png;base64," + result_b64 + "' alt='result'></div>" if result_b64 else "<div class='img-card'><p>处理结果</p><p style='color:#999;'>未执行处理</p></div>"}
  </div>
</div>
<div class="section">
  <h2>质量指标</h2>
  <table>
    <tr><th>平均灰度</th><td>{mean_val}</td><th>标准差</th><td>{std_val}</td></tr>
    <tr><th>MSE</th><td>{mse_val}</td><th>PSNR</th><td>{psnr_val}</td></tr>
    <tr><th>SSIM</th><td colspan="3">{ssim_val}</td></tr>
  </table>
</div>
<div class="footer">数字图像处理综合实验系统 &copy; 2026</div>
</body>
</html>"""

        Path(path).write_text(html, encoding="utf-8")
        self.statusBar().showMessage(f"报告已保存: {path}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
