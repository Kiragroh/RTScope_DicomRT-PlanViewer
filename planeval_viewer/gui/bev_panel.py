from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from planeval_viewer.computations.bev_geometry import (
    project_roi_outline_to_bev,
    project_roi_to_bev,
)
from planeval_viewer.dicom_io.models import (
    BeamGeometry,
    ControlPointGeometry,
    MlcLayerGeometry,
    PlanDataset,
    RoiGeometry,
)


class BevPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.plan: PlanDataset | None = None
        self.beams: list[BeamGeometry] = []
        self.current_beam: BeamGeometry | None = None
        self.target_roi: RoiGeometry | None = None
        self.target_items: list[pg.PlotDataItem] = []
        self.mlc_layer_items: list[pg.GraphicsObject] = []
        self.closed_leaf_items: list[pg.GraphicsObject] = []
        self.linac_items: list[pg.GraphicsObject] = []
        self._bev_auto_range_pending = True
        self.timer = QTimer(self)
        self.timer.setInterval(120)

        self.beam_combo = QComboBox()
        self.control_point_slider = QSlider(Qt.Orientation.Horizontal)
        self.control_point_slider.setMinimum(0)
        self.control_point_slider.setMaximum(0)
        self.play_button = QPushButton("Play video")
        self.control_point_label = QLabel("No control point")
        self.control_point_label.setObjectName("PanelSubTitle")
        self.bev_plot = pg.PlotWidget()
        self.bev_plot.setBackground("#05070b")
        self.bev_plot.setAspectLocked(True)
        self.bev_plot.showGrid(x=True, y=True, alpha=0.18)
        self.bev_plot.setLabel("bottom", "X", units="mm")
        self.bev_plot.setLabel("left", "Y", units="mm")
        self.orientation_plot = pg.PlotWidget()
        self.orientation_plot.setFixedSize(112, 112)
        self.orientation_plot.setBackground("#05070b")
        self.orientation_plot.setAspectLocked(True)
        self.orientation_plot.setXRange(-1.35, 1.35, padding=0)
        self.orientation_plot.setYRange(-1.35, 1.35, padding=0)
        self.orientation_plot.hideAxis("bottom")
        self.orientation_plot.hideAxis("left")

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Beam"))
        controls.addWidget(self.beam_combo, 1)
        controls.addWidget(self.play_button)

        cp_controls = QHBoxLayout()
        cp_controls.setContentsMargins(0, 0, 0, 0)
        cp_controls.addWidget(QLabel("CP"))
        cp_controls.addWidget(self.control_point_slider, 1)
        cp_controls.addWidget(self.control_point_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(controls)
        layout.addLayout(cp_controls)
        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.addWidget(self.bev_plot, 1)
        plot_row.addWidget(self.orientation_plot, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(plot_row, 1)

        self.beam_combo.currentIndexChanged.connect(self._select_beam)
        self.control_point_slider.valueChanged.connect(self.set_control_point_index)
        self.play_button.clicked.connect(self._toggle_playback)
        self.timer.timeout.connect(self._advance_control_point)

    def set_plan(self, plan: PlanDataset | None) -> None:
        self.plan = plan
        self.beams = list(plan.beams) if plan else []
        if plan is None:
            self.target_roi = None
        elif self.target_roi is None or plan.roi_by_name(self.target_roi.name) is None:
            self.target_roi = _first_target_roi(plan.rois)
        self.timer.stop()
        self.play_button.setText("Play video")
        self._bev_auto_range_pending = True
        preferred_index = _preferred_beam_index(self.beams)

        self.beam_combo.blockSignals(True)
        try:
            self.beam_combo.clear()
            for beam in self.beams:
                label = beam.name or f"Beam {beam.number}"
                self.beam_combo.addItem(label)
            if preferred_index >= 0:
                self.beam_combo.setCurrentIndex(preferred_index)
        finally:
            self.beam_combo.blockSignals(False)

        self._select_beam(preferred_index)

    def set_target_roi(self, roi: RoiGeometry | None) -> None:
        self.target_roi = roi
        self.set_control_point_index(self.control_point_slider.value())

    def set_control_point_index(self, index: int) -> None:
        if self.current_beam is None or not self.current_beam.control_points:
            self.control_point_label.setText("No control point")
            self.bev_plot.clear()
            self.orientation_plot.clear()
            self.linac_items = []
            return

        max_index = len(self.current_beam.control_points) - 1
        index = max(0, min(int(index), max_index))
        if self.control_point_slider.value() != index:
            self.control_point_slider.blockSignals(True)
            try:
                self.control_point_slider.setValue(index)
            finally:
                self.control_point_slider.blockSignals(False)

        cp = self.current_beam.control_points[index]
        self.control_point_label.setText(_cp_label(cp, index, len(self.current_beam.control_points)))
        self._draw_control_point(self.current_beam, cp)

    def _select_beam(self, index: int) -> None:
        if index < 0 or index >= len(self.beams):
            self.current_beam = None
            self.control_point_slider.setMaximum(0)
            self.set_control_point_index(0)
            return

        self.current_beam = self.beams[index]
        self._bev_auto_range_pending = True
        max_index = max(0, len(self.current_beam.control_points) - 1)
        self.control_point_slider.blockSignals(True)
        try:
            self.control_point_slider.setMinimum(0)
            self.control_point_slider.setMaximum(max_index)
            self.control_point_slider.setValue(0)
        finally:
            self.control_point_slider.blockSignals(False)
        self.set_control_point_index(0)

    def _toggle_playback(self) -> None:
        if self.current_beam is None or len(self.current_beam.control_points) < 2:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play video")
        else:
            self.timer.start()
            self.play_button.setText("Pause")

    def _advance_control_point(self) -> None:
        maximum = self.control_point_slider.maximum()
        if maximum <= 0:
            return
        next_index = self.control_point_slider.value() + 1
        if next_index > maximum:
            next_index = 0
        self.control_point_slider.setValue(next_index)

    def _draw_control_point(self, beam: BeamGeometry, cp: ControlPointGeometry) -> None:
        previous_range = None
        if not self._bev_auto_range_pending:
            previous_range = [axis[:] for axis in self.bev_plot.getViewBox().viewRange()]
        self.bev_plot.clear()
        self.target_items = []
        self.mlc_layer_items = []
        self.closed_leaf_items = []
        self._draw_linac_orientation(cp)
        if cp.jaws_x is not None and cp.jaws_y is not None:
            self._plot_rectangle(cp.jaws_x, cp.jaws_y, "#8ab4f8", 2)

        if not cp.has_mlc:
            self._draw_target_projection(cp)
            self._restore_or_auto_range(previous_range)
            return

        for layer_index, layer in enumerate(_display_mlc_layers(beam, cp)):
            self._plot_mlc_layer(cp, layer, layer_index)
        self._draw_target_projection(cp)
        self._restore_or_auto_range(previous_range)

    def _restore_or_auto_range(self, previous_range: list[list[float]] | None) -> None:
        if previous_range is not None:
            self.bev_plot.getViewBox().setRange(
                xRange=previous_range[0],
                yRange=previous_range[1],
                padding=0,
            )
            return
        self.bev_plot.autoRange(padding=0.12)
        self._bev_auto_range_pending = False

    def _plot_mlc_layer(
        self,
        cp: ControlPointGeometry,
        layer: MlcLayerGeometry,
        layer_index: int,
    ) -> None:
        leaf_edges = _layer_leaf_edges(layer, cp)
        left_edge_x: list[float] = []
        left_edge_y: list[float] = []
        right_edge_x: list[float] = []
        right_edge_y: list[float] = []
        left_color, right_color, material = _layer_colors(layer_index)

        for index in range(layer.leaf_count):
            y0 = leaf_edges[index]
            y1 = leaf_edges[index + 1]
            x1 = layer.mlc_x1[index]
            x2 = layer.mlc_x2[index]
            self._plot_leaf_bank(cp, x1, x2, y0, y1, material)
            left_edge_x.extend([x1, x1, np.nan])
            left_edge_y.extend([y0, y1, np.nan])
            right_edge_x.extend([x2, x2, np.nan])
            right_edge_y.extend([y0, y1, np.nan])
            if x1 >= x2:
                marker = self.bev_plot.plot(
                    [(x1 + x2) / 2.0],
                    [(y0 + y1) / 2.0],
                    pen=None,
                    symbol="x",
                    symbolSize=7,
                    symbolBrush=pg.mkBrush("#f87171"),
                    symbolPen=pg.mkPen("#fecaca", width=1.5),
                )
                self.closed_leaf_items.append(marker)

        left_item = self.bev_plot.plot(
            left_edge_x,
            left_edge_y,
            pen=pg.mkPen(left_color, width=2),
        )
        right_item = self.bev_plot.plot(
            right_edge_x,
            right_edge_y,
            pen=pg.mkPen(right_color, width=2),
        )
        self.mlc_layer_items.extend([left_item, right_item])

    def _plot_leaf_bank(
        self,
        cp: ControlPointGeometry,
        x1: float,
        x2: float,
        y0: float,
        y1: float,
        material_rgba: tuple[int, int, int, int] = (51, 65, 85, 185),
    ) -> None:
        jaw_left, jaw_right = cp.jaws_x if cp.jaws_x is not None else (-200.0, 200.0)
        jaw_left, jaw_right = sorted((float(jaw_left), float(jaw_right)))
        left_tip = _clamp(float(x1), jaw_left, jaw_right)
        right_tip = _clamp(float(x2), jaw_left, jaw_right)
        pen = pg.mkPen("#1f2937", width=0.8)
        brush = pg.mkBrush(*material_rgba)

        if left_tip > jaw_left:
            item = pg.BarGraphItem(
                x0=[jaw_left],
                x1=[left_tip],
                y0=[y0],
                y1=[y1],
                pen=pen,
                brush=brush,
            )
            self.bev_plot.addItem(item)
            self.mlc_layer_items.append(item)
        if right_tip < jaw_right:
            item = pg.BarGraphItem(
                x0=[right_tip],
                x1=[jaw_right],
                y0=[y0],
                y1=[y1],
                pen=pen,
                brush=brush,
            )
            self.bev_plot.addItem(item)
            self.mlc_layer_items.append(item)

    def _plot_rectangle(
        self,
        jaws_x: tuple[float, float],
        jaws_y: tuple[float, float],
        color: str,
        width: int,
    ) -> None:
        x0, x1 = jaws_x
        y0, y1 = jaws_y
        self.bev_plot.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            pen=pg.mkPen(color, width=width),
        )

    def _draw_target_projection(self, cp: ControlPointGeometry) -> None:
        if self.target_roi is None:
            return
        outline = project_roi_outline_to_bev(self.target_roi, cp)
        if outline is not None:
            x, y = outline
            item = self.bev_plot.plot(
                x,
                y,
                pen=pg.mkPen("#ef4444", width=3),
            )
            self.target_items.append(item)
            return
        for x, y in project_roi_to_bev(self.target_roi, cp):
            x, y = _closed_xy(x, y)
            item = self.bev_plot.plot(
                x,
                y,
                pen=pg.mkPen("#ef4444", width=2),
            )
            self.target_items.append(item)

    def _draw_linac_orientation(self, cp: ControlPointGeometry) -> None:
        self.orientation_plot.clear()
        self.linac_items = []
        theta = np.linspace(0.0, 2.0 * np.pi, 96)
        circle = self.orientation_plot.plot(
            np.cos(theta),
            np.sin(theta),
            pen=pg.mkPen("#475569", width=1),
        )
        angle = np.deg2rad(cp.gantry_angle or 0.0)
        x = float(np.sin(angle))
        y = float(np.cos(angle))
        arm = self.orientation_plot.plot([0.0, x], [0.0, y], pen=pg.mkPen("#f97316", width=3))
        head = self.orientation_plot.plot(
            [x],
            [y],
            pen=None,
            symbol="s",
            symbolSize=13,
            symbolBrush=pg.mkBrush("#f97316"),
            symbolPen=pg.mkPen("#fed7aa", width=1),
        )
        patient = self.orientation_plot.plot(
            [0.0],
            [0.0],
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#facc15"),
            symbolPen=pg.mkPen("#fef08a", width=1),
        )
        self.linac_items = [circle, arm, head, patient]


def _leaf_edges(beam: BeamGeometry, cp: ControlPointGeometry) -> np.ndarray:
    if len(beam.leaf_boundaries) >= cp.leaf_count + 1:
        return np.array(beam.leaf_boundaries[: cp.leaf_count + 1], dtype=float)
    if cp.jaws_y is not None:
        return np.linspace(cp.jaws_y[0], cp.jaws_y[1], cp.leaf_count + 1)
    return np.arange(cp.leaf_count + 1, dtype=float)


def _display_mlc_layers(
    beam: BeamGeometry,
    cp: ControlPointGeometry,
) -> list[MlcLayerGeometry]:
    if cp.mlc_layers:
        return list(cp.mlc_layers)
    return [
        MlcLayerGeometry(
            device_type="MLC",
            leaf_boundaries=beam.leaf_boundaries,
            mlc_x1=cp.mlc_x1,
            mlc_x2=cp.mlc_x2,
        )
    ]


def _layer_leaf_edges(layer: MlcLayerGeometry, cp: ControlPointGeometry) -> np.ndarray:
    if len(layer.leaf_boundaries) >= layer.leaf_count + 1:
        return np.array(layer.leaf_boundaries[: layer.leaf_count + 1], dtype=float)
    if cp.jaws_y is not None:
        return np.linspace(cp.jaws_y[0], cp.jaws_y[1], layer.leaf_count + 1)
    return np.arange(layer.leaf_count + 1, dtype=float)


def _layer_colors(index: int) -> tuple[str, str, tuple[int, int, int, int]]:
    palette = (
        ("#f97316", "#22c55e", (51, 65, 85, 185)),
        ("#38bdf8", "#c084fc", (76, 29, 149, 115)),
        ("#facc15", "#fb7185", (30, 64, 175, 95)),
    )
    return palette[index % len(palette)]


def _closed_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return x, y
    if np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1]):
        return x, y
    return np.append(x, x[0]), np.append(y, y[0])


def _preferred_beam_index(beams: list[BeamGeometry]) -> int:
    if not beams:
        return -1
    ranked = sorted(
        enumerate(beams),
        key=lambda item: (
            any(cp.has_mlc for cp in item[1].control_points),
            len(item[1].control_points),
            item[1].meterset or 0.0,
        ),
        reverse=True,
    )
    return ranked[0][0]


def _first_target_roi(rois: list[RoiGeometry]) -> RoiGeometry | None:
    for roi in rois:
        name = roi.name.upper()
        if any(marker in name for marker in ("PTV", "CTV", "GTV", "ITV")):
            return roi
    return rois[0] if rois else None


def _cp_label(cp: ControlPointGeometry, index: int, count: int) -> str:
    angle = "" if cp.gantry_angle is None else f"  Gantry={cp.gantry_angle:.1f}"
    collimator = "" if cp.collimator_angle is None else f"  Coll={cp.collimator_angle:.1f}"
    mu = f"  MUw={cp.meterset_weight:.3f}"
    return f"CP {index + 1}/{count}{angle}{collimator}{mu}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
