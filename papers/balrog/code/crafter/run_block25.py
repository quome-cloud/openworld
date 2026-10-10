"""25-episode untouched-seed robustness block (code-frozen; md5s in
results_v2/RUN_LOG.md). Seeds 11001-11025, memoryless clean protocol."""
import json
import os
import time

import run_suite

seeds = list(range(12001, 12026))
outdir = os.path.join(run_suite.RESULTS, 'block25')
os.makedirs(outdir, exist_ok=True)
results = []
for i, seed in enumerate(seeds):
    path = os.path.join(outdir, f'ep_{i:02d}_seed{seed}.json')
    if os.path.exists(path):
        results.append(json.load(open(path)))
        continue
    r = run_suite.run_episode(seed, condition='block25')
    json.dump(r, open(path, 'w'), indent=1)
    results.append(r)
    print(time.strftime('%H:%M:%S'),
          f"block25 ep{i} seed={seed} -> {r['score']:.0f}/22 "
          f"({100*r['progression']:.1f}%) steps={r['steps']} "
          f"died={r['died']} cause={r['death_cause']}", flush=True)
summary = run_suite.summarize(results, 'block25_untouched_frozen')
json.dump(summary, open(os.path.join(run_suite.RESULTS,
          'summary_block25.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items()
                  if k != 'death_causes'}, indent=1))
