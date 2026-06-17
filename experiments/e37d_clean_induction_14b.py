"""E37d - the E37c clean repro-gated induction, but with qwen2.5:14b as the inducer.

qwen2.5:7b reached repro=1.0 on 0/3 replicates (E37c), so we couldn't get a clean
"correctly-induced code extrapolates to ~1.0" data point. 14b is a stronger inducer;
this run reuses E37c's machinery verbatim (branch-covering examples, repro=1.0 gate,
held-out reachable train-region) and only swaps the model. Output -> e37d_*_14b.json.
"""

import e37c_clean_induction as m

m.MODEL = "qwen2.5:14b"   # patched before main(); induce_gated/require_ollama read the module global
m.main()

# main() wrote e37c_clean_induction.json; preserve E37c's and save this under the 14b name
src = m.RESULTS_DIR / "e37c_clean_induction.json"
dst = m.RESULTS_DIR / "e37d_clean_induction_14b.json"
if src.exists():
    data = src.read_text().replace('"e37c_clean_induction"', '"e37d_clean_induction_14b"')
    dst.write_text(data)
    print(f"\n[e37d] saved -> {dst.name}")
