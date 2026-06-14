import numpy as np
from pydicom.dataset import Dataset

import planeval_viewer.dicom_io.loader as loader_module
from planeval_viewer.dicom_io.loader import load_plan_folder, load_plan_variants
from planeval_viewer.dicom_io.models import CtVolume, PlanDataset, RoiGeometry


def test_plan_dataset_exposes_roi_names():
    plan = PlanDataset(
        ct=CtVolume(
            voxels=np.zeros((1, 2, 2)),
            z_positions=[0.0],
            pixel_spacing=(1.0, 1.0),
            origin_xy=(0.0, 0.0),
        ),
        rois=[
            RoiGeometry(
                number=1,
                name="Auge links",
                color="#800080",
                contours_by_z={},
            )
        ],
        dose=None,
        plan_info={},
        warnings=[],
    )

    assert plan.roi_names == ["Auge links"]


def test_load_plan_variants_creates_one_dataset_per_rtplan(monkeypatch, tmp_path):
    plan_a = Dataset()
    plan_a.Modality = "RTPLAN"
    plan_a.RTPlanLabel = "Plan A"
    plan_b = Dataset()
    plan_b.Modality = "RTPLAN"
    plan_b.RTPlanLabel = "Plan B"
    ct = Dataset()
    ct.Modality = "CT"
    struct = Dataset()
    struct.Modality = "RTSTRUCT"
    dose = Dataset()
    dose.Modality = "RTDOSE"

    paths = {
        "ct": tmp_path / "ct.dcm",
        "struct": tmp_path / "struct.dcm",
        "dose": tmp_path / "dose.dcm",
        "plan_a": tmp_path / "plan_a.dcm",
        "plan_b": tmp_path / "plan_b.dcm",
    }

    monkeypatch.setattr(
        loader_module,
        "_read_dicom_headers",
        lambda _folder: [
            (paths["ct"], ct),
            (paths["struct"], struct),
            (paths["dose"], dose),
            (paths["plan_a"], plan_a),
            (paths["plan_b"], plan_b),
        ],
    )
    monkeypatch.setattr(loader_module, "_load_ct", lambda _paths, _warnings: None)
    monkeypatch.setattr(loader_module, "_load_dose", lambda _paths, _warnings: None)
    monkeypatch.setattr(loader_module, "_load_rtstruct", lambda _paths, _warnings: [])
    monkeypatch.setattr(
        loader_module,
        "_load_plan_info",
        lambda plan_paths, _warnings: {"plan_label": plan_paths[0].stem},
    )
    monkeypatch.setattr(loader_module, "_load_beams", lambda _paths, _warnings: [])

    plans = load_plan_variants(tmp_path)

    assert [plan.plan_info["plan_label"] for plan in plans] == ["plan_a", "plan_b"]
    assert [plan.plan_info["source_plan_path"] for plan in plans] == [
        str(paths["plan_a"]),
        str(paths["plan_b"]),
    ]


def test_load_plan_variants_allows_image_structure_dose_without_rtplan(monkeypatch, tmp_path):
    ct = Dataset()
    ct.Modality = "CT"
    struct = Dataset()
    struct.Modality = "RTSTRUCT"
    dose = Dataset()
    dose.Modality = "RTDOSE"
    paths = {
        "ct": tmp_path / "ct.dcm",
        "struct": tmp_path / "struct.dcm",
        "dose": tmp_path / "dose.dcm",
    }
    loaded = {}

    monkeypatch.setattr(
        loader_module,
        "_read_dicom_headers",
        lambda _folder: [
            (paths["ct"], ct),
            (paths["struct"], struct),
            (paths["dose"], dose),
        ],
    )
    def record_ct(paths_, _warnings):
        loaded["ct"] = paths_
        return None

    def record_dose(paths_, _warnings):
        loaded["dose"] = paths_
        return None

    def record_struct(paths_, _warnings):
        loaded["struct"] = paths_
        return []

    monkeypatch.setattr(loader_module, "_load_ct", record_ct)
    monkeypatch.setattr(loader_module, "_load_dose", record_dose)
    monkeypatch.setattr(loader_module, "_load_rtstruct", record_struct)
    monkeypatch.setattr(loader_module, "_load_plan_info", lambda _paths, _warnings: {})
    monkeypatch.setattr(loader_module, "_load_beams", lambda _paths, _warnings: [])

    plans = load_plan_variants(tmp_path)

    assert len(plans) == 1
    assert plans[0].plan_info["plan_label"] == "Image / structure set"
    assert loaded == {
        "ct": [paths["ct"]],
        "dose": [paths["dose"]],
        "struct": [paths["struct"]],
    }


def test_load_plan_folder_returns_first_plan_variant(monkeypatch, tmp_path):
    first = PlanDataset(ct=None, rois=[], dose=None, plan_info={"plan_label": "First"})
    second = PlanDataset(ct=None, rois=[], dose=None, plan_info={"plan_label": "Second"})

    monkeypatch.setattr(loader_module, "load_plan_variants", lambda _folder: [first, second])

    assert load_plan_folder(tmp_path) is first
