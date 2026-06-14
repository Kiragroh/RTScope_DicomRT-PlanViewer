import numpy as np

from planeval_viewer.computations.dvh import DvhCurve, compute_dvh_from_samples


def test_dvh_statistics_from_samples():
    dvh = compute_dvh_from_samples(
        roi_name="Target",
        dose_values_gy=np.array([1.0, 2.0, 3.0, 4.0]),
        voxel_volume_cc=0.5,
        bin_width_gy=1.0,
    )

    assert dvh.volume_cc == 2.0
    assert dvh.dmin == 1.0
    assert dvh.dmax == 4.0
    assert dvh.dmean == 2.5
    assert dvh.dose_at_volume_pct(50) == 3.0
    assert dvh.volume_pct_at_dose(3.0) == 50.0
    assert dvh.volume_cc_at_dose(3.0) == 1.0


def test_empty_dvh_returns_not_computable_values():
    dvh = DvhCurve.empty("Empty")

    assert dvh.volume_cc == 0.0
    assert dvh.dose_at_volume_pct(95) is None
    assert dvh.volume_pct_at_dose(1.0) == 0.0
