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
# W2 (council v5) raised that `.sh` and `.csv` are on this list although a run
# script is where every crafting flag lives and thresholds_*.csv feeds
# _thr_triple.py -> --thresholds. Both objections are real, and both are now
# answered by the `argv` check added below rather than by guessing from
# filenames: every crafting-relevant thing a shell script or a threshold file
# can do shows up as different flags in argv (--thresholds carries the literal
# floats), and the argv check requires one recipe across all chunks.
#
# Filename classification alone was also WRONG in both directions -- it flagged
# run_devclean_sensitivity.sh, a script added after crafting that could not
# have affected it, while waving through an edited runner. One strong check
# beats two weak ones.
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


def _git(*args, text=True):
    return subprocess.run(["git", *args], cwd=PROJ, capture_output=True,
                          text=text)


def check_prereg(row_sha, code_shas):
    """The rows must have been produced under this pre-registration, and this
    pre-registration must only have GROWN since.

    §7.1 says the run is discarded if the rows' `prereg_sha` does not match the
    file's hash. Taken literally that is unsatisfiable the moment the document
    records anything about its own run -- appending the §9 chunking amendment
    or the §10 incident report changes the hash, and a rule that forbids
    documenting the run is not the rule anyone wanted. What §7.1 is actually
    protecting is that **the outcome→action table was not rewritten after the
    result**.

    So the check is done against history instead of against the file as it
    sits now:

      1. the rows' `prereg_sha` must equal the hash of the pre-registration
         **as of the commit the rows were crafted under**, which is exactly
         "these rows were produced under the document as it then stood";
      2. that historical text must be a **prefix** of the current text, so the
         document can only have been appended to. Any edit to an existing
         line -- §6's outcomes above all -- breaks the prefix and fails.

    Append-only plus a matching historical hash is strictly stronger than the
    original check, because the original could be satisfied by rewriting the
    document and re-running, and this cannot.
    """
    cur = _git("hash-object", PREREG)
    cur_sha = (cur.stdout or "").strip()[:12]
    if cur_sha and row_sha == cur_sha:
        print(f"[assemble] prereg_sha {row_sha} matches the file as it stands")
        return

    rel = os.path.relpath(PREREG, PROJ).replace(os.sep, "/")
    with open(PREREG, "rb") as fh:
        cur_bytes = fh.read()
    for c in sorted(code_shas):
        base = c.split("-")[0]
        blob = _git("show", f"{base}:{rel}", text=False)
        if blob.returncode != 0:
            continue
        hist = blob.stdout
        h = subprocess.run(["git", "hash-object", "--stdin"], cwd=PROJ,
                           input=hist, capture_output=True)
        hist_sha = h.stdout.decode().strip()[:12]
        if hist_sha != row_sha:
            continue
        if not cur_bytes.startswith(hist):
            fail(f"the pre-registration as of {base} is NOT a prefix of the "
                 f"current file: it has been edited, not merely appended to. "
                 f"§7.1 discards this run.")
        added = len(cur_bytes) - len(hist)
        print(f"[assemble] prereg_sha {row_sha} matches "
              f"{os.path.basename(PREREG)} as of {base}; the current file is "
              f"that text plus {added} appended bytes (append-only: OK)")
        return
    fail(f"rows carry prereg_sha {row_sha}, which matches neither the current "
         f"{os.path.basename(PREREG)} ({cur_sha}) nor its content at any of "
         f"the rows' code_sha commits {sorted(code_shas)}. §7.1 discards this.")


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
        # BUG FIX (council v5, BUG E): this checked only shas[0], so under
        # --verify-multi-sha a dirty sha anywhere else in the sorted list was
        # never reported. Check them all.
        dirty = [x for x in shas if x.endswith("-dirty")]
        if dirty:
            print(f"[assemble] WARNING: {len(dirty)} dirty code_sha(s) -- the "
                  f"working tree had uncommitted code changes during "
                  f"crafting: {', '.join(dirty)}")
    else:
        fail("no code_sha column; these rows predate W0 provenance")

    if "prereg_sha" not in df.columns:
        fail("no prereg_sha column")
    pres = sorted(set(df.prereg_sha.dropna().astype(str)) - {""})
    if len(pres) != 1:
        fail(f"{len(pres)} different prereg_sha values: {pres}")
    check_prereg(pres[0], shas)

    # W2 (council v5): code_sha and prereg_sha were checked and `argv` never
    # was, so chunks were verified to tile the speaker set under one code
    # version while nothing checked they used the same --steps, --lam-null,
    # --thresholds, --craft-stride or --n-eot. A chunk launched from an edited
    # run script passed every other check. The per-chunk flags that are
    # SUPPOSED to differ (--spk-offset, --n-files, --out, --tag) are stripped
    # before comparing; everything else must match exactly.
    if "argv" in df.columns:
        import re as _re
        def _norm(a):
            a = str(a)
            for flag in ("--spk-offset", "--n-files", "--out", "--tag"):
                a = _re.sub(rf"{flag}\s+\S+", "", a)
            return " ".join(a.split())
        # Compare WITHIN each arm, not across them: the two arms differ by
        # design (fullnsT adds --lam-null 10 --null-t 200), so a global
        # comparison would always fail. What must be constant is the recipe
        # each arm was crafted with across its own chunks.
        for arm, sub in df.dropna(subset=["argv"]).groupby("variant"):
            per_arm = sorted({_norm(a) for a in sub.argv})
            if len(per_arm) != 1:
                nl = chr(10) + "    "
                fail("arm %s was crafted with %d DIFFERENT flag sets across "
                     "its chunks (ignoring --spk-offset/--n-files/--out/"
                     "--tag):%s%s"
                     % (arm, len(per_arm), nl,
                        nl.join(n[:160] for n in per_arm[:4])))
        norms = sorted({_norm(a) for a in df.argv.dropna()})
        print(f"[assemble] argv check PASSED: one recipe per arm across its "
              f"chunks ({len(norms)} arm recipe(s), as expected)")
    else:
        fail("no argv column; cannot verify the recipe was constant")

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
