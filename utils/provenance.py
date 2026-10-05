"""utils/provenance.py -- which code wrote this row?  (PLAN_V3 W0)

Every row an experiment script writes gets three columns:

  code_sha    git HEAD that produced the row, suffixed "-dirty" when any
              tracked file under the code directories differed from HEAD.
  prereg_sha  content hash (`git hash-object`) of the pre-registration file
              named by --prereg, or "" when none was given.  A row can then be
              checked against the exact outcome->action table that was
              committed before it was crafted (PLAN_V3 W3-W5, §25 format).
  argv        the full command line, shell-quoted, so the row is reproducible
              without the run script that launched it.

The CLI is what experiments/skipguard.sh calls:

  python utils/provenance.py                 -> prints the current code_sha
  python utils/provenance.py --stored OUT TAG
        -> prints the code_sha values found in OUT's rows whose first column
           is TAG (joined with "|"), "MISSING" if those rows have no code_sha
           column, or "" if no row carries that tag.

Why: the run scripts' `grep -q "^$tag,"` skip-guard treated "rows exist" as
"this arm is done".  It would silently merge arms crafted by two code versions
into one CSV -- the RESULTS §25 failure mode.  With the sha stored per row the
guard can refuse instead.
"""
import csv
import os
import shlex
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Directories whose tracked files define "the code".  A change anywhere here
# marks the sha dirty; results CSVs, docs and paper text do not.
CODE_DIRS = ("experiments", "core", "models", "utils", "evaluation", "main.py")

_cache = {}


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=PROJ, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def code_sha():
    """git HEAD, plus '-dirty' if tracked code differs from it. 'nogit' if unavailable."""
    if "code_sha" in _cache:
        return _cache["code_sha"]
    head = _git("rev-parse", "--short=12", "HEAD")
    if head is None:
        sha = "nogit"
    else:
        # Only *.py / *.sh under the code dirs count; a CSV being rewritten by
        # the very script that is running must not mark its own rows dirty.
        status = _git("status", "--porcelain", "--untracked-files=no", "--", *CODE_DIRS)
        dirty = any(line[3:].rstrip().endswith((".py", ".sh"))
                    for line in (status or "").splitlines() if line.strip())
        sha = head + ("-dirty" if dirty else "")
    _cache["code_sha"] = sha
    return sha


def file_sha(path):
    """Content hash of a file as git would store it ('' if no path given)."""
    if not path:
        return ""
    if not os.path.isfile(path):
        raise SystemExit(f"[provenance] --prereg file not found: {path}")
    sha = _git("hash-object", os.path.abspath(path))
    if sha is None:
        import hashlib
        with open(path, "rb") as fh:
            sha = "sha256:" + hashlib.sha256(fh.read()).hexdigest()
    return sha[:12]


def argv_string():
    return " ".join(shlex.quote(a) for a in sys.argv)


def provenance(prereg=None):
    """The three columns to splat into every output row."""
    return dict(code_sha=code_sha(), prereg_sha=file_sha(prereg), argv=argv_string())


def add_prereg_arg(ap):
    """Attach the shared --prereg option to an argparse parser."""
    ap.add_argument("--prereg", default="",
                    help="pre-registration file (outcome->action table) committed before "
                         "this run; its content hash is stored in every row as prereg_sha")
    return ap


def stored_code_sha(out_csv, tag):
    """What skipguard.sh asks: the code_sha(s) behind the rows tagged TAG in OUT."""
    if not os.path.isfile(out_csv):
        return ""
    found = set()
    have_col = None
    with open(out_csv, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if header is None:
            return ""
        have_col = "code_sha" in header
        col = header.index("code_sha") if have_col else None
        for row in rdr:
            if not row or row[0] != tag:
                continue
            found.add(row[col] if (have_col and col < len(row)) else "MISSING")
    if not found:
        return ""
    return "|".join(sorted(found))


def _main(argv):
    if len(argv) >= 3 and argv[0] == "--stored":
        print(stored_code_sha(argv[1], argv[2]))
    elif argv:
        raise SystemExit(__doc__)
    else:
        print(code_sha())


if __name__ == "__main__":
    _main(sys.argv[1:])
