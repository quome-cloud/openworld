"""Leaderboard verification: parse the NetHack column from the RAW
balrogai.com table HTML (curl snapshot committed under results/evidence/).

Rule (operator mandate 2026-07-06): leaderboard numbers must come from
regex over the leaderboard-table `<td>` cells of the raw HTML — never from
a web-page summarizer (three summarizer fetches produced three different
column alignments that day; one caused a MiniHack/NetHack column misread,
35.0 vs the true 6.8).

Usage: python3 verify_leaderboard.py [snapshot.html]
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "results", "evidence",
                       "balrog_leaderboard_2026-07-06.html")


def parse(path):
    html = open(path, encoding="utf-8").read()
    m = re.search(r"<table[^>]*id=\"leaderboard-table-LLM\".*?</table>",
                  html, re.S)
    tab = m.group(0) if m else re.findall(r"<table.*?</table>", html, re.S)[0]
    heads = [re.sub(r"<[^>]+>|\s+", " ", h).strip()
             for h in re.findall(r"<th[^>]*>(.*?)</th>", tab, re.S)]
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", tab, re.S)[1:]:
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>|&nbsp;", " ", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
        if cells:
            rows.append(cells)
    return heads, rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    heads, rows = parse(path)
    i = heads.index("NetHack")
    ranked = []
    for r in rows:
        if len(r) > i:
            try:
                v = float(r[i].split("±")[0])
            except ValueError:
                continue
            ranked.append((v, r[i], r[0]))
    ranked.sort(reverse=True)
    print(f"headers: {heads}")
    print("NetHack column, top 5:")
    for v, cell, name in ranked[:5]:
        print(f"  {cell:12s} {name}")
    out = {"headers": heads, "rows": rows,
           "nethack_top5": [{"value": cell, "agent": name}
                            for _, cell, name in ranked[:5]]}
    outp = os.path.join(os.path.dirname(path), "leaderboard_parse.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
