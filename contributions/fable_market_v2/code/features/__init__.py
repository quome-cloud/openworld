"""Feature library registry. Each module defines FAMILY, FEATURES, compute()."""
from . import sll_sector_leadlag, peer_momentum, reversal_volume, pead_earnings_drift

MODULES = [sll_sector_leadlag, peer_momentum, reversal_volume, pead_earnings_drift]
FAMILIES = {m.FAMILY: m for m in MODULES}


def compute_all(panel) -> dict[str, dict]:
    """Returns {feature_name: {"df": DataFrame, "family": str}}."""
    out = {}
    for m in MODULES:
        for name, df in m.compute(panel).items():
            assert name in m.FEATURES
            out[name] = {"df": df, "family": m.FAMILY}
    return out


def dictionary_rows() -> list[dict]:
    """Feature dictionary for the report (name, family, one-line equation)."""
    rows = []
    for m in MODULES:
        doc = (m.__doc__ or "").strip().splitlines()[0]
        for name in m.FEATURES:
            rows.append({"feature": name, "family": m.FAMILY,
                         "module": m.__name__.split(".")[-1] + ".py",
                         "summary": doc})
    return rows
