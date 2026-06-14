from planeval_viewer.computations.complexity import aperture_area, leaf_travel_mm
from planeval_viewer.dicom_io.models import BeamGeometry, ControlPointGeometry
from planeval_viewer.dicom_io.loader import _control_point_from_dataset


def test_loader_splits_standard_mlcx_positions_into_leaf_banks():
    cp = _control_point_from_dataset(
        _cp_dataset(
            [
                ("MLCX", [-20.0, -10.0, 15.0, 25.0]),
            ]
        )
    )

    assert cp.mlc_x1 == (-20.0, -10.0)
    assert cp.mlc_x2 == (15.0, 25.0)
    assert cp.leaf_count == 2
    assert len(cp.mlc_layers) == 1
    assert cp.mlc_layers[0].device_type == "MLCX"
    assert cp.mlc_layers[0].mlc_x1 == (-20.0, -10.0)
    assert cp.mlc_layers[0].mlc_x2 == (15.0, 25.0)


def test_loader_preserves_double_stacked_halcyon_mlc_layers():
    cp = _control_point_from_dataset(
        _cp_dataset(
            [
                ("MLCX1", [-130.0, -25.0, -128.0, -12.0]),
                ("MLCX2", [22.0, 128.0, 35.0, 130.0]),
            ]
        )
    )

    assert len(cp.mlc_layers) == 2
    assert cp.mlc_layers[0].device_type == "MLCX1"
    assert cp.mlc_layers[0].mlc_x1 == (-130.0, -25.0)
    assert cp.mlc_layers[0].mlc_x2 == (-128.0, -12.0)
    assert cp.mlc_layers[1].device_type == "MLCX2"
    assert cp.mlc_layers[1].mlc_x1 == (22.0, 128.0)
    assert cp.mlc_layers[1].mlc_x2 == (35.0, 130.0)
    assert cp.mlc_x1 == (-130.0, -25.0)
    assert cp.mlc_x2 == (-128.0, -12.0)
    assert cp.leaf_count == 2


def test_control_point_has_mlc_positions():
    cp = ControlPointGeometry(
        index=3,
        meterset_weight=0.5,
        gantry_angle=180.0,
        collimator_angle=10.0,
        couch_angle=0.0,
        jaws_x=(-50.0, 50.0),
        jaws_y=(-40.0, 40.0),
        mlc_x1=(-20.0, -10.0),
        mlc_x2=(20.0, 10.0),
    )

    assert cp.has_mlc
    assert cp.leaf_count == 2


def test_aperture_area_uses_open_leaf_widths_and_jaw_height():
    cp = ControlPointGeometry(
        index=0,
        meterset_weight=0.0,
        jaws_y=(-50.0, 50.0),
        mlc_x1=(-20.0, -10.0),
        mlc_x2=(20.0, 10.0),
    )

    assert aperture_area(cp) == 3000.0


def test_leaf_travel_sums_absolute_bank_motion_between_control_points():
    first = ControlPointGeometry(
        index=0,
        meterset_weight=0.0,
        mlc_x1=(-20.0, -10.0),
        mlc_x2=(20.0, 10.0),
    )
    second = ControlPointGeometry(
        index=1,
        meterset_weight=0.5,
        mlc_x1=(-18.0, -14.0),
        mlc_x2=(22.0, 9.0),
    )

    assert leaf_travel_mm(first, second) == 9.0


def test_beam_reports_control_point_count():
    beam = BeamGeometry(number=1, name="B1", control_points=[ControlPointGeometry(index=0)])

    assert beam.control_point_count == 1


def _cp_dataset(devices):
    from pydicom.dataset import Dataset

    cp = Dataset()
    cp.ControlPointIndex = 0
    cp.CumulativeMetersetWeight = 0.0
    cp.BeamLimitingDevicePositionSequence = []
    for device_type, positions in devices:
        device = Dataset()
        device.RTBeamLimitingDeviceType = device_type
        device.LeafJawPositions = positions
        cp.BeamLimitingDevicePositionSequence.append(device)
    return cp
