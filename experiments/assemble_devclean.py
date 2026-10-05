"""
experiments/assemble_devclean.py
================================
Concatenate the #5 crafting chunks, and refuse if the union is not exactly the
31 pre-registered speakers.

This is the gate in `PREREG_05_devclean_replication.md` §9. The chunked
execution is an allowance, not a licence: the only thing that makes it
equivalent to one `--spk-offset 0 --n-files 31` run is that the chunks tile the
cross-session speaker list exactly once, with one `code_sha` and one
`prereg_sha`. All of that is checked here, and a failure aborts rather than
warns — a warning in a run script's log is a warning nobody reads.

Checks, each fatal:
  1. every chunk file exists and is non-empty
  2. the speaker union equals the 31 cross-session-usable dev-clean speakers,
     with no duplicate speaker across chunks
  3. exactly one `code_sha` across all rows (prereg §7.5)
  4. exactly one `prereg_sha`, and it is this file's current hash (§7.1)
  5. both arms are present for every speaker

Run:
  python experiments/assemble_devclean.py
Output:
  experiments/results/main_table_dc_m30_seg_bpda.csv
"""
import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

R = os.path.join(PROJ, "experiments", "results")
PREREG = os.path.join(PROJ, "paper", "PREREG_05_devclean_replication.md")
ARMS = ("full_dc_m30_seg_bpda", "fullnsT_dc_m30_seg_bpda")


def fail(msg):
    raise SystemExit(f"[assemble] REFUSING to assemble: {msg}\n"
                     f"  The chunked run is only equivalent to one unchunked "
                     f"run if the chunks tile the pre-registered speaker set "
                     f"exactly once under one code version. See "
                     f"PREREG_05_devclean_replication.md §9 and §7.")


# Paths whose contents cannot reach the crafting computation: analysis scripts,
# shell wrappers, result files, tests and prose. Anything in a diff that is NOT
# matched here causes a refusal -- including a file this list has never seen,
# because "unclassified" must fail closed rather than pass.
ANALYSIS_ONLY = (
    "experiments/results/",      # outputs
    "experiments/analyse_",      # analysis entry points
    "experiments/diag_",         # diagnostics
    "experiments/assemble_",     # this file
    "experiments/make_",         # table / page / stimulus generators
    "experiments/eval_frontend.py",
    "experiments/eval_clone_quality.py",
    "experiments/serve_listening_test.py",
    "paper/", "tests/", "listening_test/", "notebooks/", "docs/",
)
ANALYSIS_ONLY_SUFFIX = (".md", ".csv", ".sh", ".txt", ".json", ".bib", ".png")


def verify_shas_are_analysis_only(shas):
    """Accept several code_sha values ONLY if nothing between them can have
    changed the crafting computation.

    The sha is computed over `utils/provenance.CODE_DIRS`, which includes the
    whole of `experiments/` -- so committing an ANALYSIS script, or even a
    result CSV, during a long crafting run bumps it. That is what happened on
    2026-10-06 (see PREREG_05 §10): five commits landed between chunk 1 and
    chunk 2, none of them on the crafting import path.

    This function does not take that on trust. It diffs every pair of shas and
    refuses unless every changed path is classified as unable to affect
    crafting. A path the classifier does not recognise is a refusal.
    """
    import itertools
    for x, y in itertools.combinations(sorted(shas), 2):
        r = subprocess.run(["git", "diff", "--name-only", x, y], cwd=PROJ,
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"cannot diff {x}..{y}: {r.stderr.strip()[:200]}")
        changed = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        bad = [c for c in changed
               if not (c.startswith(ANALYSIS_ONLY)
                       or c.endswith(ANALYSIS_ONLY_SUFFIX))]
        if bad:
            fail("%s..%s changes %d file(s) that may affect crafting:\n    %s"
                 % (x, y, len(bad), "\n    ".join(bad[:12])))
        print(f"[assemble] {x}..{y}: {len(changed)} file(s) changed, all "
              f"analysis-only -- crafting path untouched")
    print(f"[assemble] multi-sha accepted under --verify-multi-sha: "
          f"{', '.join(shas)}")


def expected_speakers():
    from experiments.librispeech import (list_librispeech, reference_items,
                                         cross_session_split)
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "dev-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0, utt_index=0)
    ok = set(cross_session_split(items, min_sec=4.0))
    return sorted(s for s in ref if s in ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(
        R, "main_table_dc_m30_seg_bpda_c*.csv"))
    ap.add_argument("--out", default=os.path.join(
        R, "main_table_dc_m30_seg_bpda.csv"))
    ap.add_argument("--verify-multi-sha", action="store_true",
                    dest="verify_multi_sha",
                    help="allow several code_sha values, but only after "
                         "checking that every file changed between them is "
                         "analysis-only and the crafting import path is "
                         "untouched")
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        fail(f"no chunk files matching {a.pattern}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if not len(d):
            fail(f"{os.path.basename(f)} is empty")
        d["_chunk"] = os.path.basename(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # The file column is spk<id>_<chapter>_<utt>; recover the speaker.
    df["speaker"] = df.file.astype(str).str.extract(r"^spk(\d+)_")[0].astype(int)

    want = expected_speakers()
    got = sorted(df.speaker.unique())
    if got != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        fail(f"speaker union is {len(got)} speakers, expected {len(want)}.\n"
             f"  missing: {missing}\n  unexpected: {extra}")

    # No speaker may appear in two chunks: that would mean the offsets
    # overlapped and some speaker is double-weighted in every paired test.
    per = df.drop_duplicates(["speaker", "variant", "_chunk"])
    dup = (per.groupby(["speaker", "variant"])["_chunk"].nunique() > 1)
    if dup.any():
        fail(f"{int(dup.sum())} (speaker, arm) pair(s) appear in more than one "
             f"chunk: {sorted({s for s, _ in dup[dup].index})}")

    if "code_sha" in df.columns:
        shas = sorted(set(df.code_sha.dropna().astype(str)))
        if len(shas) != 1:
            if not a.verify_multi_sha:
                fail(f"{len(shas)} different code_sha values in one assembly: "
                     f"{shas}\n"
                     f"  If the difference is provably analysis-only, re-run "
                     f"with --verify-multi-sha, which CHECKS that rather than "
                     f"taking your word for it.")
            verify_shas_are_analysis_only(shas)
        if shas[0].endswith("-dirty"):
            print(f"[assemble] WARNING: code_sha {shas[0]} is dirty -- the "
                  f"working tree had uncommitted code changes during crafting.")
    else:
        fail("no code_sha column; these rows predate W0 provenance")

    cur = subprocess.run(["git", "hash-object", PREREG], cwd=PROJ,
                         capture_output=True, text=True)
    cur_sha = (cur.stdout or "").strip()[:12]
    if "prereg_sha" in df.columns:
        pres = sorted(set(df.prereg_sha.dropna().astype(str)) - {""})
        if len(pres) != 1:
            fail(f"{len(pres)} different prereg_sha values: {pres}")
        if cur_sha and pres[0] != cur_sha:
            fail(f"rows carry prereg_sha {pres[0]} but "
                 f"{os.path.basename(PREREG)} now hashes to {cur_sha}. The "
                 f"pre-registration changed after the run; §7.1 discards this.")
    else:
        fail("no prereg_sha column")

    have = set(df.variant.unique())
    if not set(ARMS) <= have:
        fail(f"expected both arms {ARMS}, found {sorted(have)}")
    for arm in ARMS:
        n = df[df.variant == arm].speaker.nunique()
        if n != len(want):
            fail(f"arm {arm} covers {n} speakers, expected {len(want)}")

    df.drop(columns=["_chunk"]).to_csv(a.out, index=False)
    print(f"[assemble] {len(files)} chunks -> {len(df)} rows, "
          f"{df.speaker.nunique()} speakers, arms {sorted(have)}")
    print(f"[assemble] code_sha {shas[0]} | prereg_sha {pres[0]} "
          f"(matches {os.path.basename(PREREG)})")
    print(f"[assemble] union check PASSED against the {len(want)} "
          f"cross-session dev-clean speakers")
    print(f"[assemble] saved {a.out}")


if __name__ == "__main__":
    main()
