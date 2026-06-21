"""E80 (tabular) - world-time compute on a VARIETY of REAL classification problems.

Each real OpenML-CC18 dataset is a WORLD (a CSV from a different real problem); a row is an
example; the label is the real class. The shared skill is generic "given a feature schema and
values, predict the class." We train on many dataset-worlds and test on STRICTLY held-out
datasets -- so a hit means the model learned transferable tabular-reasoning, not one schema.
Real data, ground-truth labels -> directly rebuts the "we generated it" critique.

Consumed by e80_common (build_worlds + CONFIG). Needs the `openml` package (installed on box).
"""

CONFIG = {
    "ladder": [2, 8, 32, 64, 128],
    "abl_n": 48,
    "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
    "cap": 80,             # rows per dataset-world
    "n_test": 40,          # held-out datasets (strict)
    "seeds": [0, 1],
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
}

MAX_FEATURES = 30
MAX_CLASSES = 12
ROWS_FETCH = 200          # rows considered per dataset
MAX_WORLDS = 250          # build up to this many real tabular worlds
MAX_CANDIDATES = 700      # OpenML datasets to attempt (many fail/dup -> skipped) (subsample of large ones)


def _fmt(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.3g}"
    except (TypeError, ValueError):
        return str(v)[:24]


def _candidate_ids():
    """Query OpenML for many real classification datasets (small/medium), not just CC18."""
    import openml
    df = openml.datasets.list_datasets(output_format="dataframe", status="active")
    keep = df[(df.NumberOfClasses >= 2) & (df.NumberOfClasses <= MAX_CLASSES)
              & (df.NumberOfFeatures >= 2) & (df.NumberOfFeatures <= MAX_FEATURES + 1)
              & (df.NumberOfInstances >= 120) & (df.NumberOfInstances <= 20000)
              & (df.NumberOfMissingValues == 0)]
    keep = keep.sort_values("name").drop_duplicates("name")     # one version per dataset name
    return list(dict.fromkeys(keep.did.astype(int).tolist()))[:MAX_CANDIDATES]


def build_worlds():
    import openml
    worlds = {}
    for did in _candidate_ids():
        if len(worlds) >= MAX_WORLDS:
            break
        try:
            ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                             download_features_meta_data=False)
            if not ds.default_target_attribute:
                continue
            X, y, _, cols = ds.get_data(target=ds.default_target_attribute)
        except Exception as e:  # noqa: BLE001
            print(f"  [tabular] skip did {did}: {repr(e)[:60]}", flush=True)
            continue
        if X is None or y is None or X.shape[1] > MAX_FEATURES:
            continue
        classes = sorted({str(v) for v in y.tolist() if str(v) != "nan"})
        if not (2 <= len(classes) <= MAX_CLASSES):
            continue
        feat_cols = list(X.columns)
        rows = []
        n = min(len(X), ROWS_FETCH)
        for i in range(n):
            lab = str(y.iloc[i])
            if lab == "nan":
                continue
            feats = ", ".join(f"{c}={_fmt(X.iloc[i][c])}" for c in feat_cols)
            prompt = (f"Classify this record from the '{ds.name}' dataset.\n"
                      f"Features: {feats}\nPossible classes: {', '.join(classes)}\n"
                      "Answer with ONLY the exact class label.")
            rows.append({"prompt": prompt, "label": lab})
        if len(rows) >= 60 and len({r["label"] for r in rows}) >= 2:
            worlds[ds.name] = {"classes": classes, "rows": rows}
            print(f"  [tabular] world '{ds.name}': {len(rows)} rows, {len(classes)} classes",
                  flush=True)
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    print(f"tabular worlds: {len(w)}")
