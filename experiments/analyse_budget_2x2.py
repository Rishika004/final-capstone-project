"""
experiments/analyse_budget_2x2.py
=================================
Did the mismatch claim survive the rebase, or was it a property of the budget?

WHAT WENT WRONG, AND WHY THIS SCRIPT IS NOT analyse_mismatch.py
---------------------------------------------------------------
The published claim -- "the pass-through constraint buys protection when the
attacker's purifier is one you did not assume" -- was measured at the LOUD budget
(m30_f15) with POOLED d-vector crafting, on dev-clean / purifier v3 /
same-session thresholds / 20 speakers.

The 2026-09-20 rebase moved it to test-clean / v7 / cross-session / 29 speakers,
which was the intent, but ALSO silently to the TIGHT budget (m20_f20) and to
--seg-partials crafting, which was not. It came back with the sign flipped: 4 of
12 cells significant instead of 7, every one on the d-vector, every one POSITIVE
(constraint protects WORSE). WavLM effects shrank 4-5x and stopped surviving Holm.

A rebase that changes the treatment is not a rebase. This script reads the two
budgets on the SAME corrected base, scored against the SAME calibrations.

CAVEAT, ADDED AFTER THE FIRST READ (2026-09-22): this is NOT a 2x2. The m20 arms
were crafted WITH --seg-partials and the m30 arms WITHOUT, so the two rows are
the DIAGONAL of a budget x crafting table. What a difference between them proves
is that {budget, crafting} together matter -- not which one. The name of this
file overstates it; it is kept so the RESULTS cross-references resolve. The
missing off-diagonal cell (m30 + seg) is run_rebase_mismatch_m30_seg.sh, read
by analyse_mismatch_final.py once it exists.

  m30 holds, m20 fails  -> the effect is REAL and BUDGET-CONDITIONAL. The paper
                           must say "at the loud budget", not state it flatly.
                           That is a weaker claim than the draft currently makes.
  both fail             -> the effect was a property of the old evaluation base.
                           Drop the claim. Rows 4 and 5 of the headline table go.
  both hold             -> the m20 result was the seg-partials reallocation, and
                           that is then the thing to investigate.

ON MULTIPLICITY, WHICH THIS DESIGN MAKES SHARPER
------------------------------------------------
Testing two budgets and reporting whichever survives is how a null becomes a
finding. So BOTH corrections are printed:

  within-budget (m=12)  comparable to the published analysis, and what the claim
                        would have been had we only ever run that budget.
  across-budget (m=24)  the honest correction for having looked at two, given we
                        are here precisely because the first one failed.

If a cell survives at m=12 but not m=24, say so in the paper rather than quoting
the m=12 number. This is a post-hoc rescue of a failed confirmation and should be
labelled as one -- the RESULTS entry gets written that way or not at all.

Statistics are on the continuous decision margin (cos_after_purify - threshold,
negative = protected), paired by file, one test per encoder. Never pooled across
encoders: 3 encoders x 29 files is not n=87.

Run:  python experiments/analyse_budget_2x2.py
"""
import sys
import io
import os

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.analyse_mismatch import ENC, load, holm, paired, row

R = os.path.join(PROJ, "experiments", "results")

# (label, plain, nsT, full, fullnsT) per budget.
BUDGETS = [
    ("m30_f15  LOUD  (the budget the claim was published at)",
     "plain_tc_m30", "nsT_tc_m30", "full_tc_m30", "fullnsT_tc_m30"),
    ("m20_f20  TIGHT (the budget the rebase silently used)",
     "plain_tc_m20", "nsT_tc_m20", "full_tc_m20", "fullnsT_tc_m20"),
]

CELLS = [("DiffWave-LibriSpeech t=1", "protect_foreign_both_libri_t1.csv"),
         ("DiffWave-LibriSpeech t=3", "protect_foreign_both_libri_t3.csv"),
         ("DiffWave-SC09 t=1", "protect_foreign_both_sc09_t1.csv"),
         ("DiffWave-SC09 t=3", "protect_foreign_both_sc09_t3.csv")]


def mismatch_family(budget):
    """The 12 cells of block (c) for one budget: 2 purifiers x 2 strengths x 3 enc.

    Foreign files carry their OWN calibration and are not re-thresholded here
    (load(channel=None)); passing a channel would apply our purifier's threshold
    to a DiffWave-purified row, which is fix S2 run backwards.
    """
    _, _, _, full, fullnst = budget
    fam = []
    for lbl, fn in CELLS:
        d = load(os.path.join(R, fn))
        for e in ENC:
            fam.append((f"{lbl}|{e}", paired(d, full, fullnst, enc=e)))
    return fam


def control_family(budget):
    """Block (a): no purifier. The constraint must LOSE here, or the mismatch
    framing is wrong -- a merely sturdier perturbation would win everywhere."""
    _, _, _, full, fullnst = budget
    d = load(os.path.join(R, "control_nopurifier_both.csv"))
    return [(e, paired(d, full, fullnst, enc=e)) for e in ENC]


def summarise(fam, key="p_adj"):
    live = [r for _, r in fam if r is not None and r.get(key) is not None]
    sig = [r for r in live if r[key] < 0.05]
    return len(sig), len(live), sum(1 for r in sig if r["delta"] < 0)


def main():
    print("=" * 78)
    print("BLOCK (c)  MISMATCHED PURIFIER -- the positive claim, both budgets")
    print("=" * 78)
    print("  negative delta = EOT+constraint protects BETTER than EOT alone.")
    print("  'better' counts speakers, not cells.\n")

    fams = {}
    for b in BUDGETS:
        fam = mismatch_family(b)
        if all(r is None for _, r in fam):
            print(f"\n### {b[0]}\n  (not available -- run "
                  f"experiments/run_rebase_mismatch_m30.sh first)")
            continue
        holm(fam)                       # within-budget, m=12
        fams[b[0]] = fam
        print(f"\n### {b[0]}")
        cur = None
        for lbl, r in fam:
            head, e = lbl.split("|")
            if head != cur:
                print(f"\n  {head}")
                cur = head
            row(f"    {e}", r, width=30)

    if len(fams) < 2:
        print(f"\n{len(fams)} of 2 budgets present; the 2x2 cannot be read yet.")
        return

    # Across-budget correction. Recomputed from raw p, because holm() above
    # overwrote p_adj with the within-budget value.
    joint = [(f"{bl} :: {lbl}", r) for bl, fam in fams.items() for lbl, r in fam]
    for _, r in joint:
        if r is not None:
            r.pop("p_adj", None)
    holm(joint)
    across = {}
    for bl in fams:
        across[bl] = [(l.split(" :: ", 1)[1], r) for l, r in joint
                      if l.startswith(bl)]

    print("\n" + "=" * 78)
    print("BLOCK (a)  NO PURIFIER -- the control that makes the reading work")
    print("=" * 78)
    print("  The constraint must LOSE here (positive delta). If it wins in both")
    print("  blocks the result is generic robustness, not mismatch insurance.\n")
    for b in BUDGETS:
        fam = control_family(b)
        if all(r is None for _, r in fam):
            continue
        holm(fam)
        print(f"  {b[0]}")
        for lbl, r in fam:
            row(f"    {lbl}", r, width=30)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  {'budget':<12}{'sig (m=12)':>12}{'of which help':>15}"
          f"{'sig (m=24)':>12}{'of which help':>15}")
    for bl in fams:
        s12, n12, h12 = summarise(fams[bl])
        s24, _, h24 = summarise(across[bl])
        print(f"  {bl.split()[0]:<12}{f'{s12}/{n12}':>12}{h12:>15}"
              f"{f'{s24}/{n12}':>12}{h24:>15}")
    print("
  NOT A CLEAN 2x2: m20 rows were crafted with --seg-partials, m30 rows")
    print("  without. A difference between them is budget AND/OR crafting. See")
    print("  RESULTS section 26c for the run that separates them.")
    print("\n  'of which help' counts significant cells with a NEGATIVE delta --")
    print("  the only ones that support the claim. A significant cell with a")
    print("  POSITIVE delta is evidence AGAINST it and must not be counted as a")
    print("  hit. The old headline read '7 of 12 after Holm' without this split.")
    print("\n  Published claim for reference: 7 of 12, all negative, at m30_f15 on")
    print("  dev-clean / v3 / same-session / n=20 / pooled d-vector crafting.")


if __name__ == "__main__":
    main()
