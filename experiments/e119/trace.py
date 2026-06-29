"""E119 reproducibility: provenance capture + a runs.jsonl-style per-run trace logger.
The SLM arm is not bit-reproducible (sampling + Metal); this pins everything that IS fixed and
records every run so a draw can be audited and anchored to its replayable action sequence."""
import json, hashlib, sys, platform


def prompt_digest(text):
    b = text.encode("utf-8")
    return {"chars": len(text), "lines": text.count("\n") + 1,
            "sha256": hashlib.sha256(b).hexdigest(), "approx_tokens": max(1, len(text) // 4)}


def _versions():
    v = {"python": platform.python_version()}
    for mod in ("numpy", "arc_agi", "arcengine"):
        try:
            import importlib.metadata as m
            v[mod] = m.version(mod.replace("_", "-"))
        except Exception:
            v[mod] = None
    return v


def provenance(model, options, seeds, budget, digest=None):
    """Reproducibility metadata for the results `env` block. `digest` is best-effort (None ok)."""
    return {"model": model, "options": dict(options or {}), "seeds": list(seeds),
            "budget": dict(budget or {}), "digest": digest, "versions": _versions()}


def log_run(path, record):
    """Append one run record as a JSON line (runs.jsonl-style)."""
    import pathlib
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")


def ollama_digest(model):
    """Best-effort: return the local Ollama digest string for `model`, or None on any failure.
    Never raises; never blocks longer than the default urllib timeout (~few seconds)."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"name": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("digest") or data.get("details", {}).get("digest")
    except Exception:
        return None
