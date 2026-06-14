import os
from pathlib import Path

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from PySide6.QtWidgets import QApplication

import planeval_viewer.gui.dicom_tag_panel as dicom_tag_panel_module
from planeval_viewer.gui.dicom_tag_panel import DicomTagPanel


def test_dicom_tag_panel_expands_nested_sequences_and_filters_values():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = DicomTagPanel()
    ds = Dataset()
    ds.Modality = "RTPLAN"
    beam = Dataset()
    beam.BeamName = "Arc 1"
    cp = Dataset()
    cp.ControlPointIndex = 0
    beam.ControlPointSequence = Sequence([cp])
    ds.BeamSequence = Sequence([beam])

    panel._populate_dataset(ds)

    assert panel.tree.topLevelItemCount() >= 2
    beam_item = _find_item(panel.tree.invisibleRootItem(), "BeamSequence")
    assert beam_item is not None
    assert _find_item(beam_item, "ControlPointSequence") is not None
    assert _find_item(beam_item, "ControlPointIndex") is not None

    panel.search_edit.setText("ControlPointIndex")
    assert not _find_item(beam_item, "ControlPointIndex").isHidden()
    assert beam_item.isExpanded()

    panel.close()
    app.processEvents()


def test_dicom_tag_panel_groups_ct_slices_as_series(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    panel = DicomTagPanel()
    ct1 = tmp_path / "ct1.dcm"
    ct2 = tmp_path / "ct2.dcm"
    plan = tmp_path / "plan.dcm"
    for path in (ct1, ct2, plan):
        path.write_bytes(b"DICOM")

    def fake_dcmread(path, stop_before_pixels=True, force=True):
        name = Path(path).name
        ds = Dataset()
        if name.startswith("ct"):
            ds.Modality = "CT"
            ds.SeriesInstanceUID = "1.2.3"
            ds.SeriesDescription = "Planning CT"
            ds.SOPInstanceUID = "1.2.3.1" if name == "ct1.dcm" else "1.2.3.2"
            ds.InstanceNumber = 1 if name == "ct1.dcm" else 2
            ds.ImagePositionPatient = [0.0, 0.0, float(ds.InstanceNumber)]
            ds.Rows = 512
            ds.Columns = 512
            ds.PixelSpacing = [1.0, 1.0]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            return ds
        ds.Modality = "RTPLAN"
        ds.SOPInstanceUID = "9.9.9"
        return ds

    monkeypatch.setattr(dicom_tag_panel_module.pydicom, "dcmread", fake_dcmread)

    panel.set_folder(tmp_path)

    labels = [panel.file_combo.itemText(index) for index in range(panel.file_combo.count())]
    assert any("CT series" in label and "2 slices" in label for label in labels)
    assert sum(1 for label in labels if label.startswith("CT series")) == 1

    ct_index = next(index for index, label in enumerate(labels) if label.startswith("CT series"))
    panel.file_combo.setCurrentIndex(ct_index)

    assert _find_item(panel.tree.invisibleRootItem(), "NumberOfSeriesRelatedInstances") is not None
    assert _find_item(panel.tree.invisibleRootItem(), "ReferencedImageSequence") is not None

    panel.close()
    app.processEvents()


def _find_item(parent, text):
    for index in range(parent.childCount()):
        child = parent.child(index)
        if any(text in child.text(column) for column in range(4)):
            return child
        nested = _find_item(child, text)
        if nested is not None:
            return nested
    return None
