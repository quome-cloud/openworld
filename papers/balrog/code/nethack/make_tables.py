"""Summarize results JSONs into the report tables (stdout markdown)."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def ep_row(e):
    return (f"| {e['episode']} | {e['seed']} | {e.get('role')} | "
            f"{e['progression']*100:.2f} | {e.get('highest_achievement')} | "
            f"{e['depth_max']} | {e['xplvl_max']} | {e['steps']} | "
            f"{e['end_reason']} |")


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        RESULTS, "nethack_results.json")
    doc = json.load(open(fn))
    if "episodes" in doc:
        eps = doc["episodes"]
        print(f"### {doc.get('label')} (seed base {doc.get('seed_base')})")
        print("| ep | seed | role | prog% | best | depth | xp | steps | end |")
        print("|---|---|---|---|---|---|---|---|---|")
        for e in sorted(eps, key=lambda x: x["episode"]):
            print(ep_row(e))
        print(f"\n**score: {doc['score']:.2f}**")
    else:
        for p in doc["passes"]:
            print(f"### memory pass {p['pass']} (seeds {p['seed_base']}+)")
            print("| ep | seed | role | prog% | best | depth | xp | steps | end |")
            print("|---|---|---|---|---|---|---|---|---|")
            for e in sorted(p["episodes"], key=lambda x: x["episode"]):
                print(ep_row(e))
            print(f"\npass score: {p['score']:.2f}\n")


if __name__ == "__main__":
    main()
