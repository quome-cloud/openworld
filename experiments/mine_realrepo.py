"""Mine real per-commit world-model traces from open-source git repositories.

For each commit we record the repository's STATE, measured directly from the git
tree (`git ls-tree` / blob-cached `git cat-file`), and the ACTION, measured from
the commit's diff against its first parent (`git diff`). State and action come
from INDEPENDENT git queries - we never build state by summing diffs - so the
laws E45 induces (e.g. next_loc = loc + insertions - deletions) are genuine
empirical claims about git, not identities-by-construction.

Output: experiments/data/realrepo/<repo>.csv, committed so the analysis
(e45_real_repo_induction.py) reruns fully offline. Regenerating the CSVs needs
network (a shallow clone per repo).

Usage:  python experiments/mine_realrepo.py            # all repos
        python experiments/mine_realrepo.py requests   # one repo
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPOS = {
    "requests": "https://github.com/psf/requests.git",
    "flask": "https://github.com/pallets/flask.git",
    "tqdm": "https://github.com/tqdm/tqdm.git",
}
WINDOW = 600                                   # most-recent commits per repo
OUT = Path(__file__).resolve().parent / "data" / "realrepo"

FIELDS = ["sha", "files", "py_files", "test_files", "py_loc",
          "a_added", "a_deleted", "a_py_added", "a_py_deleted",
          "a_py_ins", "a_py_del"]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True, check=True).stdout


def is_test(path):
    return ("tests/" in path or "/test_" in path or path.startswith("test_")
            or path.endswith("_test.py"))


def tree_blobs(repo, sha):
    """[(blobsha, path)] for every file in the tree at `sha`."""
    out = git(repo, "ls-tree", "-r", sha)
    blobs = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _mode, _type, blobsha = meta.split()
        blobs.append((blobsha, path))
    return blobs


def blob_line_counts(repo, shas):
    """{blobsha: newline_count} for text blobs, via one batched cat-file pass.
    Binary blobs (containing a NUL byte) are omitted."""
    shas = list(shas)
    if not shas:
        return {}
    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    payload = ("\n".join(shas) + "\n").encode()
    data, _ = proc.communicate(payload)
    counts = {}
    i = 0
    for sha in shas:
        nl = data.index(b"\n", i)
        header = data[i:nl].decode()
        i = nl + 1
        parts = header.split()
        if len(parts) != 3 or parts[1] != "blob":   # missing/invalid object
            continue
        size = int(parts[2])
        content = data[i:i + size]
        i += size + 1                                # skip content + trailing \n
        counts[sha] = 0 if b"\x00" in content else content.count(b"\n")
    return counts


def measure_state(repo, sha, loc_cache):
    blobs = tree_blobs(repo, sha)
    py = [(b, p) for b, p in blobs if p.endswith(".py")]
    missing = [b for b, _ in py if b not in loc_cache]
    loc_cache.update(blob_line_counts(repo, missing))
    return {
        "files": len(blobs),
        "py_files": len(py),
        "test_files": sum(1 for _, p in blobs if is_test(p)),
        "py_loc": sum(loc_cache.get(b, 0) for b, _ in py),
    }


def measure_action(repo, sha):
    """The diff of `sha` against its first parent. Plain diff (no rename
    detection) so renames appear as add+delete, which keeps the counting laws
    robust. Returns added/deleted file counts and .py line churn."""
    parent = git(repo, "rev-parse", f"{sha}^").strip()
    a = {"a_added": 0, "a_deleted": 0, "a_py_added": 0, "a_py_deleted": 0,
         "a_py_ins": 0, "a_py_del": 0}
    for line in git(repo, "diff", "--name-status", parent, sha).splitlines():
        st, path = line.split("\t", 1)
        path = path.split("\t")[-1]
        if st.startswith("A"):
            a["a_added"] += 1
            a["a_py_added"] += path.endswith(".py")
        elif st.startswith("D"):
            a["a_deleted"] += 1
            a["a_py_deleted"] += path.endswith(".py")
    for line in git(repo, "diff", "--numstat", parent, sha, "--", "*.py").splitlines():
        ins, dele, _path = line.split("\t", 2)
        if ins != "-":
            a["a_py_ins"] += int(ins)
            a["a_py_del"] += int(dele)
    return a


def mine_repo(name, url):
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        repo = str(Path(tmp) / name)
        print(f"[{name}] cloning (depth {WINDOW + 5}) ...")
        subprocess.run(["git", "clone", "--quiet", f"--depth={WINDOW + 5}",
                        url, repo], check=True)
        shas = git(repo, "rev-list", "--first-parent", "-n", str(WINDOW),
                   "HEAD").split()
        shas.reverse()                              # chronological (oldest first)
        # commits whose first parent exists in the shallow clone (need it for the
        # action); the oldest shaved-off boundary commit is dropped.
        loc_cache = {}
        rows = []
        for j, sha in enumerate(shas):
            try:
                action = measure_action(repo, sha)
            except subprocess.CalledProcessError:
                continue                            # parent beyond shallow graft
            state = measure_state(repo, sha, loc_cache)
            rows.append({"sha": sha, **state, **action})
            if (j + 1) % 100 == 0:
                print(f"[{name}] {j + 1}/{len(shas)} commits")
        path = OUT / f"{name}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[{name}] wrote {len(rows)} commits -> {path}")


def main():
    which = sys.argv[1:] or list(REPOS)
    for name in which:
        mine_repo(name, REPOS[name])


if __name__ == "__main__":
    main()
