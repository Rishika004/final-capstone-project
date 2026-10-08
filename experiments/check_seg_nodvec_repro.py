"""Does HEAD still craft what recipe-v1 crafted on the seg held-out path?

RESULTS 47e makes one crafting arm mandatory: `full_m30_seg_nodvec`, the
unconstrained counterpart of the `fullnsT_m30_seg_nodvec` arm that has been on
disk since 2026-09-20.  The pair (fullnsT - full) is then the seg-budget
version of Control A.

The hazard is RESULTS 25's: the existing arm was crafted before the W0
provenance columns, before `--craft-seed` was a flag, and before the W1
selection fixes landed (all off by default, pinned by a golden file).  If HEAD
is not numerically identical on this path, the pair would compare two code
versions and the comparison would be worthless -- and nothing on disk would
say so.

The golden file (tests/test_selection.py, 15 tests) pins the *objective* and
the *ship-iterate selector*.  It does not pin the whole crafting loop.  So this
script re-crafts ONE speaker of the EXISTING arm from HEAD, with the flags the
runner used, and compares against the stored row.

Agreement -> the new arm may be crafted from HEAD and paired with the old one,
and that is recorded.  Disagreement -> both arms must be re-crafted from HEAD,
at twice the GPU cost, and 47e's "one arm" estimate was wrong.

Usage:
    python experiments/check_seg_nodvec_repro.py --stage craft
    python experiments/check_seg_nodvec_repro.py --stage compare
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.csvio import read_csv  # noqa: E402

R = ROOT / "experiments" / "results"
STORED = R / "main_table_tc_seg.csv"
PROBE = R / "repro_seg_nodvec_probe.csv"
TAG_STORED = "fullnsT_m30_seg_nodvec"
TAG_PROBE = "repro_fullnsT_m30_seg_nodvec"

# Exactly run_p0_segments.sh's COMMON + craft B, with --n-files 1.
THRFILE = R / "thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv"

# Columns that must agree bit-for-bit if the crafting path is unchanged.
# craft_s (wall clock) and the provenance columns are expected to differ.
NUMERIC = ["cos_before", "cos_after_purify", "cos_after_max", "cos_ctrl_unprot",
           "pesq", "stoi", "snr_db"]
EXACT = ["encoder", "eval_t", "threshold"]


def thr_triple():
    out = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "_thr_triple.py"), str(THRFILE)],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def craft(out=PROBE, tag=TAG_PROBE):
    if not THRFILE.exists():
        sys.exit(f"FATAL: {THRFILE} missing")
    thr = thr_triple()
    print(f"[repro] thresholds: {thr}", flush=True)
    cmd = [
        sys.executable, str(ROOT / "experiments" / "exp_purifier_aware.py"),
        "--dataset", "librispeech", "--subset", "test-clean", "--cross-session",
        "--spk-offset", "0", "--utt-index", "0", "--n-files", "1",
        "--steps", "200", "--purifier-ckpt", "checkpoints/purifier_v7.pt",
        "--sampler", "ddim", "--thresholds", thr, "--eval-t", "50,200",
        "--thr-aware", "--weights", "1,1,1", "--seg-partials",
        "--craft-encoders", "no-dvec",
        "--variant", "full", "--vocoder", "phase",
        "--margin-db", "30", "--snr-floor", "15",
        "--lam-null", "10", "--null-t", "200",
        "--out", str(out), "--tag", tag,
    ]
    print("[repro] " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[repro] craft exit {rc}", flush=True)
    return rc


# -------------------------------------------------------- determinism control
# The stage-1 comparison cannot distinguish "the code changed" from "the
# crafting loop is not run-to-run deterministic".  Without this control a
# disagreement would be misread as a code difference -- and every
# one-variable comparison in the project (RESULTS 30, 47) would inherit the
# same ambiguity.  So: craft the SAME flags twice from HEAD.
PROBE2 = R / "repro_seg_nodvec_probe2.csv"
TAG_PROBE2 = "repro2_fullnsT_m30_seg_nodvec"


def compare_two(path_a, tag_a, path_b, tag_b, label_a, label_b):
    a_df = read_csv(path_a)
    b_df = read_csv(path_b)
    a_df = a_df[a_df["variant"] == tag_a]
    b_df = b_df[b_df["variant"] == tag_b]
    if len(a_df) == 0 or len(b_df) == 0:
        sys.exit(f"FATAL: no rows ({label_a}: {len(a_df)}, {label_b}: {len(b_df)})")
    f = sorted(set(b_df["file"]))[0]
    o = a_df[a_df["file"] == f].sort_values(EXACT).reset_index(drop=True)
    n = b_df[b_df["file"] == f].sort_values(EXACT).reset_index(drop=True)
    if len(o) == 0:
        sys.exit(f"FATAL: {label_a} has no rows for {f!r}")
    print(f"\n{label_a}  vs  {label_b}     file = {f}")
    print(f"{'encoder':<22}{'t':>5}{'column':>20}{'a':>12}{'b':>12}{'|d|':>12}")
    worst, ndiff = 0.0, 0
    for i in range(min(len(o), len(n))):
        ro, rn = o.iloc[i], n.iloc[i]
        for c in NUMERIC:
            if c not in o.columns or c not in n.columns:
                continue
            try:
                x, y = float(ro[c]), float(rn[c])
            except (TypeError, ValueError):
                continue
            d = abs(x - y)
            worst = max(worst, d)
            if d > 1e-6:
                ndiff += 1
                print(f"{str(ro['encoder']):<22}{str(ro['eval_t']):>5}{c:>20}"
                      f"{x:>12.6f}{y:>12.6f}{d:>12.2e}")
    print(f"  worst |difference| = {worst:.3e}   columns differing: {ndiff}")
    return worst, ndiff


def compare():
    old = read_csv(STORED)
    new = read_csv(PROBE)
    old = old[old["variant"] == TAG_STORED]
    new = new[new["variant"] == TAG_PROBE]
    if len(new) == 0:
        sys.exit("FATAL: probe produced no rows")

    f = sorted(set(new["file"]))[0]
    o = old[old["file"] == f]
    n = new[new["file"] == f]
    if len(o) == 0:
        sys.exit(f"FATAL: stored arm has no rows for file {f!r}; "
                 f"stored files: {sorted(set(old['file']))[:3]}")

    key = EXACT
    o = o.sort_values(key).reset_index(drop=True)
    n = n.sort_values(key).reset_index(drop=True)
    print(f"\nfile = {f}   rows: stored {len(o)}, probe {len(n)}\n")

    worst = 0.0
    bad = []
    print(f"{'encoder':<22}{'t':>5}{'column':>20}{'stored':>12}{'probe':>12}{'|d|':>12}")
    for i in range(min(len(o), len(n))):
        ro, rn = o.iloc[i], n.iloc[i]
        for c in NUMERIC:
            if c not in o.columns or c not in n.columns:
                continue
            try:
                a, b = float(ro[c]), float(rn[c])
            except (TypeError, ValueError):
                continue
            d = abs(a - b)
            worst = max(worst, d)
            if d > 1e-6:
                bad.append((ro["encoder"], ro["eval_t"], c, a, b, d))
            print(f"{str(ro['encoder']):<22}{str(ro['eval_t']):>5}{c:>20}"
                  f"{a:>12.6f}{b:>12.6f}{d:>12.2e}")

    print(f"\nworst |difference| over {len(NUMERIC)} columns x {len(o)} rows: {worst:.3e}")
    if worst <= 1e-6:
        print("\nVERDICT: REPRODUCES. HEAD crafts what recipe-v1 crafted on this\n"
              "path, so `full_m30_seg_nodvec` may be crafted from HEAD and paired\n"
              "with the stored `fullnsT_m30_seg_nodvec`. One arm, as 47e budgeted.")
        return 0
    print(f"\nVERDICT: DOES NOT REPRODUCE ({len(bad)} columns differ).")
    print("Both arms must be re-crafted from HEAD; 47e's one-arm estimate is wrong.")
    for row in bad[:12]:
        print("   ", row)
    return 1


def verdict():
    """Both comparisons together, which is the only way to read either."""
    print("=" * 92)
    print("A. HEAD vs HEAD -- is the crafting loop run-to-run deterministic?")
    print("=" * 92)
    w_self, n_self = compare_two(PROBE, TAG_PROBE, PROBE2, TAG_PROBE2,
                                 "HEAD run 1", "HEAD run 2")
    print()
    print("=" * 92)
    print("B. recipe-v1 (stored, 2026-09-20) vs HEAD")
    print("=" * 92)
    w_ver, n_ver = compare_two(STORED, TAG_STORED, PROBE, TAG_PROBE,
                               "stored v1", "HEAD run 1")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  HEAD vs HEAD      worst {w_self:.4f}  ({n_self} columns differ)")
    print(f"  stored v1 vs HEAD worst {w_ver:.4f}  ({n_ver} columns differ)")
    if w_self <= 1e-6 and w_ver <= 1e-6:
        print("\n  DETERMINISTIC AND UNCHANGED. One arm suffices, as 47e budgeted.")
        return 0
    if w_self <= 1e-6 and w_ver > 1e-6:
        print("\n  DETERMINISTIC BUT CHANGED. HEAD does not craft what v1 crafted,")
        print("  so the stored arm may NOT be paired with a HEAD-crafted arm.")
        print("  Both arms must be re-crafted from HEAD: 47e's one-arm estimate")
        print("  is wrong and the seg control costs ~2 GPU-h, not ~1.")
        return 1
    if w_self > 1e-6 and w_ver <= w_self * 2:
        print("\n  NOT DETERMINISTIC. HEAD does not reproduce ITSELF, and the")
        print("  stored-vs-HEAD difference is the same size as HEAD-vs-HEAD.")
        print("  So the stage-B difference carries NO information about code")
        print("  version -- but every one-variable crafting comparison in this")
        print("  project inherits this run-to-run term, and nothing has")
        print("  quantified it. Record it as a floor and craft BOTH arms in one")
        print("  run so the pair shares its draw.")
        return 2
    print("\n  NOT DETERMINISTIC, AND the version difference exceeds the")
    print("  run-to-run term. Both effects are present; craft both arms.")
    return 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["craft", "craft2", "compare", "verdict"])
    a = ap.parse_args()
    if a.stage == "craft":
        sys.exit(craft())
    if a.stage == "craft2":
        sys.exit(craft(PROBE2, TAG_PROBE2))
    if a.stage == "compare":
        sys.exit(compare())
    sys.exit(verdict())
