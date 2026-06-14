import os

import numpy as np
from PySide6.QtWidgets import QApplication

from planeval_viewer.dicom_io.models import PlanDataset, RoiGeometry
from planeval_viewer.gui.details_panel import DetailsPanel
from planeval_viewer.gui.qa_panel import QAPanel
from planeval_viewer.plan_targets import select_default_target_name


def test_default_target_prefers_plan_target_and_ptv_before_ctv():
    plan = PlanDataset(
        ct=None,
        rois=[
            RoiGeometry(number=1, name="CTV BWK 8-12", color="#00aaff"),
            RoiGeometry(number=2, name="PTV1 BWK 8-12", color="#ff0000"),
        ],
        dose=None,
        plan_info={"target_prescriptions": [{"description": "PTV1 BWK 8-12"}]},
    )

    assert select_default_target_name(plan) == "PTV1 BWK 8-12"

    plan.plan_info = {}
    assert select_default_target_name(plan) == "PTV1 BWK 8-12"


def test_qa_target_combo_can_override_target_and_details_show_selection():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = QAPanel()
    details = DetailsPanel()
    plan = PlanDataset(
        ct=None,
        rois=[
            RoiGeometry(number=1, name="CTV", color="#00aaff"),
            RoiGeometry(number=2, name="PTV", color="#ff0000"),
            RoiGeometry(number=3, name="OAR", color="#00ff00"),
        ],
        dose=None,
        plan_info={"target_roi_name": "PTV", "prescription_dose_gy": 30.0},
    )

    panel.set_plan(plan)
    details.set_plan(plan)
    details.set_target_name(panel.selected_target_name())

    assert panel.selected_target_name() == "PTV"
    assert "Target: PTV" in details.plan_text.toPlainText()

    panel.target_combo.setCurrentText("CTV")
    details.set_target_name(panel.selected_target_name())

    assert panel.selected_target_name() == "CTV"
    assert "Target: CTV" in details.plan_text.toPlainText()

    panel.close()
    details.close()
    app.processEvents()
