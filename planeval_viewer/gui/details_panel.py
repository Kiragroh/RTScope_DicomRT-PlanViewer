from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from planeval_viewer.dicom_io.models import PlanDataset
from planeval_viewer.refdb.matching import RoiLookup


class DetailsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.plan: PlanDataset | None = None
        self.target_name = ""
        self.plan_title = QLabel("No plan loaded")
        self.plan_title.setStyleSheet("font-weight: 700; font-size: 12pt;")
        self.plan_text = QTextEdit()
        self.plan_text.setReadOnly(True)
        self.roi_title = QLabel("Selected ROI")
        self.roi_title.setStyleSheet("font-weight: 700; font-size: 12pt;")
        self.roi_text = QTextEdit()
        self.roi_text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.plan_title)
        layout.addWidget(self.plan_text, 1)
        layout.addWidget(self.roi_title)
        layout.addWidget(self.roi_text, 2)

    def set_plan(self, plan: PlanDataset | None) -> None:
        self.plan = plan
        if plan is None:
            self.plan_title.setText("No plan loaded")
            self.plan_text.clear()
            self.roi_text.clear()
            return

        info = plan.plan_info
        title = info.get("plan_label") or info.get("plan_name") or "Loaded RT plan"
        self.plan_title.setText(str(title))
        lines = [
            f"Patient: {info.get('patient_name', '')}",
            f"Patient ID: {info.get('patient_id', '')}",
            f"Fractions: {info.get('number_of_fractions', '')}",
            f"Prescription: {info.get('prescription_dose_gy', '')} Gy",
            f"Dose/Fx: {info.get('dose_per_fraction_gy', '')} Gy",
            f"Target: {self.target_name}",
            f"ROIs: {len(plan.rois)}",
        ]
        if plan.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in plan.warnings)
        self.plan_text.setPlainText("\n".join(lines))

    def set_target_name(self, name: str) -> None:
        self.target_name = name
        self.set_plan(self.plan)

    def set_roi(self, source_name: str, lookup: RoiLookup | None) -> None:
        self.roi_title.setText(source_name or "Selected ROI")
        if not source_name:
            self.roi_text.clear()
            return
        if lookup is None:
            self.roi_text.setPlainText("No RefDB lookup result yet.")
            return

        lines = [
            f"Status: {lookup.status}",
            f"Matched: {lookup.matched_name}",
            f"Reference: {lookup.reference_name}",
            f"Side: {lookup.side}",
            f"Color: {lookup.color}",
        ]
        if lookup.aliases:
            lines.append(f"Aliases: {', '.join(lookup.aliases)}")
        if lookup.error:
            lines.append("")
            lines.append(f"Not found: {lookup.error}")

        result = lookup.result
        if result and result.constraint_tables:
            lines.append("")
            lines.append("Constraint tables:")
            for table in result.constraint_tables[:20]:
                count = len(table.constraints)
                fraction = _format_fraction_range(table.fx_min, table.fx_max)
                marker = " bilateral" if table.is_bilateral else ""
                lines.append(f"- {table.name}{marker}: {count} constraints{fraction}")
            if len(result.constraint_tables) > 20:
                lines.append(f"... {len(result.constraint_tables) - 20} more")
        self.roi_text.setPlainText("\n".join(lines))


def _format_fraction_range(fx_min: float | None, fx_max: float | None) -> str:
    if fx_min is None and fx_max is None:
        return ""
    if fx_min == fx_max:
        return f", {fx_min:g} Fx"
    left = "" if fx_min is None else f"{fx_min:g}"
    right = "" if fx_max is None else f"{fx_max:g}"
    return f", {left}-{right} Fx"
