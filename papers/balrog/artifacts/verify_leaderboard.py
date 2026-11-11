"""Verify the BALROG leaderboard BabaIsAI column directly from the HTML.

Context: an earlier pass cited 75.7% (Gemini-3.1-Pro-Thinking) as the
BabaIsAI SOTA -- that figure is actually Gemini-3.1-Pro-Thinking's
TextWorld column. This script re-derives the true BabaIsAI column from the
raw table so any reviewer can re-verify:

    curl -s https://balrogai.com > leaderboard.html
    python3 verify_leaderboard.py leaderboard.html

Output on 2026-07-07:
    Gemini-3.1-Pro             BabaIsAI 90.0 +- 2.7   (TextWorld 66.5)
    Gemini-3-Pro               BabaIsAI 88.3 +- 2.9   (TextWorld 60.2)
    Gemini-3.1-Pro-Thinking    BabaIsAI 83.3 +- 3.4   (TextWorld 75.7)  <- the misread source
    Gemini-3-Flash             BabaIsAI 73.3 +- 4.0   (TextWorld 50.2)
=> BabaIsAI SOTA = 90.0 +- 2.7 (Gemini-3.1-Pro).

Sibling script: ../../minihack/verify_leaderboard.py (PR #211) does the
same cross-check for the MiniHack column, which was misread in the
opposite direction (an early pass reported the BabaIsAI figure as MiniHack
SOTA). Same root cause both times: reading the wrong column off the table.
"""

import re
import sys


def main(path):
    html = open(path).read()
    table = re.findall(r"<table[^>]*>(.*?)</table>", html, re.S)[0]
    header = [re.sub(r"<[^>]+>", "", h).strip()
              for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)]
    baba_i = header.index("BabaIsAI")
    tw_i = header.index("TextWorld")
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)[1:]:
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
        if len(cells) == len(header):
            rows.append((cells[0].split()[-1], cells[baba_i], cells[tw_i]))
    rows.sort(key=lambda x: -float(x[1].split(" ±")[0]))
    for name, baba, tw in rows[:6]:
        print(f"{name:30s} BabaIsAI {baba:12s} TextWorld {tw}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "leaderboard.html")
