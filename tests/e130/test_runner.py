import numpy as np
from experiments import e130_shu_cycle as R


def test_validate_theorems_reports_separation_and_contraction():
    m = R.validate_theorems(np.random.default_rng(0))
    assert m["expert_error_100"] < m["expert_error_1"] / 5.0       # Thm 4.4
    assert m["amateur_trials_mean"] > m["expert_consultations"]    # separation
    assert m["final_tension"] < 1e-6 and m["rho"] < 1.0            # Thm 4.6
