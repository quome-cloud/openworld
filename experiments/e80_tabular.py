"""E80 (tabular) - world-time compute on a VARIETY of REAL classification problems.

Each real OpenML-CC18 dataset is a WORLD (a CSV from a different real problem); a row is an
example; the label is the real class. The shared skill is generic "given a feature schema and
values, predict the class." We train on many dataset-worlds and test on STRICTLY held-out
datasets -- so a hit means the model learned transferable tabular-reasoning, not one schema.
Real data, ground-truth labels -> directly rebuts the "we generated it" critique.

Consumed by e80_common (build_worlds + CONFIG). Needs the `openml` package (installed on box).
"""

CONFIG = {
    "ladder": [2, 4, 8, 16, 32],
    "abl_n": 16,
    "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
    "cap": 100,            # rows per dataset-world
    "n_test": 15,          # held-out datasets (strict)
    "seeds": [0, 1],
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
}

MAX_FEATURES = 30
MAX_CLASSES = 12
ROWS_FETCH = 400          # rows considered per dataset (subsample of large ones)


def _fmt(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.3g}"
    except (TypeError, ValueError):
        return str(v)[:24]


def build_worlds():
    import openml
    worlds = {}
    suite = openml.study.get_suite("OpenML-CC18")
    for tid in suite.tasks:
        try:
            task = openml.tasks.get_task(tid, download_data=False, download_qualities=False)
            ds = openml.datasets.get_dataset(task.dataset_id, download_data=True,
                                             download_qualities=False, download_features_meta_data=False)
            X, y, _, cols = ds.get_data(target=ds.default_target_attribute)
        except Exception as e:  # noqa: BLE001
            print(f"  [tabular] skip task {tid}: {repr(e)[:80]}", flush=True)
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
