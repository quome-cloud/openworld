"""Adapter checkpointing for the fine-tuning harness: save a trained LoRA to GCS right after
training and reload it (skipping retrain) before eval/inference. So a preemption during the
inference phase doesn't discard the expensive training, and adapters become reusable artifacts.

Usage in a train/eval loop:
    from _adapter_ckpt import load_or_train
    load_or_train(model, tag, bucket, reset_adapter,
                  lambda: fit(model, tok, rows, steps, seed))
    # ... then eval as usual ...
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpt"


def _gcs(*args):
    subprocess.run(["gcloud", "storage", *args], check=False)


def save_adapter(model, tag, bucket=""):
    d = CKPT / tag
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(d))            # saves the ACTIVE LoRA adapter
    if bucket:
        _gcs("cp", "-r", str(d), f"{bucket.rstrip('/')}/ckpt/{tag}/")


def _restore_from_gcs(tag, bucket):
    d = CKPT / tag
    if bucket and not (d / "adapter_config.json").exists():
        CKPT.mkdir(parents=True, exist_ok=True)
        _gcs("cp", "-r", f"{bucket.rstrip('/')}/ckpt/{tag}", str(CKPT))
    return d


def load_adapter(model, tag, bucket, reset_fn):
    """If a saved adapter for `tag` exists (local or GCS), load its weights into a fresh active
    adapter and return True; else return False (caller should train). Never raises -- on any
    failure it falls back to training."""
    try:
        from peft import load_peft_weights, set_peft_model_state_dict
        d = _restore_from_gcs(tag, bucket)
        if (d / "adapter_config.json").exists():
            reset_fn(model)                  # fresh active adapter to load into
            set_peft_model_state_dict(model, load_peft_weights(str(d)))
            print(f"[ckpt] resumed adapter '{tag}' (skipped training)", flush=True)
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[ckpt] resume failed for '{tag}' ({e}); will train", flush=True)
    return False


def load_or_train(model, tag, bucket, reset_fn, train_fn):
    """Resume a saved adapter if present, else reset + train + checkpoint."""
    if load_adapter(model, tag, bucket, reset_fn):
        return
    reset_fn(model)
    train_fn()
    save_adapter(model, tag, bucket)
