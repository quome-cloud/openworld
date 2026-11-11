"""Verify the BALROG leaderboard MiniHack column directly from the HTML.

Context: an earlier summarizer pass misread the leaderboard and reported
90.0 as "MiniHack SOTA" — that value is the *BabaIsAI* column of
Gemini-3.1-Pro. This script re-derives the true column from the raw table
so any reviewer can re-verify:

    curl -s https://balrogai.com > leaderboard.html
    python3 verify_leaderboard.py leaderboard.html

Output on 2026-07-06:
    Gemini-3-Pro               MiniHack 40.0 +- 7.7   (BabaIsAI 88.3)
    Gemini-3.1-Pro             MiniHack 35.0 +- 7.5   (BabaIsAI 90.0)  <- the misread source
    Claude-Opus-4.5-Thinking   MiniHack 30.0 +- 7.2
    Gemini-3-Flash             MiniHack 30.0 +- 7.2
=> MiniHack SOTA = 40.0 +- 7.7 (Gemini-3-Pro).
"""

import re
import sys


def main(path):
    html = open(path).read()
    table = re.findall(r"<table[^>]*>(.*?)</table>", html, re.S)[0]
    header = [re.sub(r"<[^>]+>", "", h).strip()
              for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)]
    mh_i = header.index("MiniHack")
    baba_i = header.index("BabaIsAI")
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)[1:]:
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
        if len(cells) == len(header):
            rows.append((cells[0].split()[-1], cells[baba_i], cells[mh_i]))
    rows.sort(key=lambda x: -float(x[2].split(" ±")[0]))
    for name, baba, mh in rows[:6]:
        print(f"{name:40s} MiniHack {mh:12s} BabaIsAI {baba}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "leaderboard.html")
