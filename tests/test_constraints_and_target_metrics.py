import numpy as np

from planeval_viewer.computations.constraints import compute_constraint_metric, constraint_passes
from planeval_viewer.computations.dvh import compute_dvh_from_samples
from planeval_viewer.computations.target_metrics import compute_paddick_metrics_from_masks
from planeval_viewer.refdb.models import ConstraintRow


def test_constraint_metric_supports_dmax_dmean_dcc_and_vgy():
    dvh = compute_dvh_from_samples(
        "OAR",
        np.array([1.0, 2.0, 3.0, 4.0]),
        voxel_volume_cc=0.5,
        bin_width_gy=1.0,
    )

    assert compute_constraint_metric(dvh, ConstraintRow(metric="Dmax", unit="Gy")) == 4.0
    assert compute_constraint_metric(dvh, ConstraintRow(metric="Dmean", unit="Gy")) == 2.5
    assert compute_constraint_metric(dvh, ConstraintRow(metric="D0.5cc", unit="Gy")) == 4.0
    assert compute_constraint_metric(dvh, ConstraintRow(metric="V3Gy", unit="%")) == 50.0
    assert compute_constraint_metric(dvh, ConstraintRow(metric="V3Gy", unit="cc")) == 1.0


def test_constraint_passes_uses_optimal_then_maximal_limit():
    row = ConstraintRow(metric="Dmax", unit="Gy", comparator="<=", limit_optimal=3.0, limit_maximal=4.0)

    assert constraint_passes(2.5, row) == "optimal"
    assert constraint_passes(3.5, row) == "acceptable"
    assert constraint_passes(4.5, row) == "fail"


def test_paddick_metrics_from_masks():
    target = np.array([True, True, False, False])
    dose = np.array([10.0, 5.0, 10.0, 0.0])

    metrics = compute_paddick_metrics_from_masks(
        target_mask=target,
        dose_gy=dose,
        prescription_gy=10.0,
        voxel_volume_cc=1.0,
    )

    assert metrics.target_volume_cc == 2.0
    assert metrics.prescription_isodose_volume_cc == 2.0
    assert metrics.target_covered_volume_cc == 1.0
    assert metrics.coverage == 0.5
    assert metrics.selectivity == 0.5
    assert metrics.paddick_ci == 0.25
    assert metrics.gradient_index == 1.5
    assert metrics.d2_gy == 9.9
    assert metrics.d98_gy == 5.1
    assert metrics.homogeneity_index == 0.52
