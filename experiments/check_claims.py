"""
experiments/check_claims.py
===========================
Grep the draft for wordings `FRAME_v6` §4 and `FRAME_v7` forbid.

WHY A SCRIPT AND NOT A READ-THROUGH
-----------------------------------
The claims ledger now has ~25 rows, and most of them forbid a phrase that is
the NATURAL thing to write. "Equivalent", "replicated", "null", "deployment",
"robust to purification" -- every one of them is what a tired author types at
2 a.m., and every one has a specific result behind it saying why it is wrong.
A human re-read catches them until it doesn't.

This runs in a second and fails loudly. It is a lint, not a judge: a hit is
not proof of an error, because some of these phrases are legitimate inside a
"not this" table or a quotation. Every hit is printed with its line so the
author decides; the exit code is non-zero only for hits outside the known-safe
contexts.

WHAT TO LINT, AND WHAT NOT TO
-----------------------------
Lint the PAPER documents: draft_v6.md, FRAME_v7.md, THREAT_MODEL.md. Those are
what a reviewer reads and what the ledger governs.

Do NOT act on hits in `RESULTS_ICASSP.md`. It is an append-only lab notebook,
most of it written before the rules existed, and §§1-20 are on a superseded
base that the paper may not quote from at all (`FRAME_v6` base rule). Rewriting
history to satisfy a lint would destroy the record of what was believed when.
Running the lint over it is informative -- it shows which old sections would
mislead if quoted -- and nothing more.

Run:
    python experiments/check_claims.py
    python experiments/check_claims.py paper/draft_v6.md paper/FRAME_v7.md
"""
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = [os.path.join(PROJ, "paper", "draft_v6.md")]

# (pattern, why it is forbidden, the section that forbids it)
RULES = [
    (r"\bequivalen(t|ce)\b"
     r"(?!.{0,120}(fail|tested as|claim it is|margin|support equivalence"
     r"|only at|bound))",
     "the pre-specified TOST at D=0.01 FAILED; say 'bounded by 0.032'",
     "RESULTS 51"),
    (r"\breplicat(ed|es|ion)\b(?!.{0,60}(out-of-sample replication|, not))",
     "two samples read separately; write 'directionally reproduced'",
     "FRAME_v6 4 / RESULTS 45"),
    (r"every effect (is )?null|a reported null|\bclean null\b",
     "46 is INCONCLUSIVE: 8/8 cells below their MDE, median power 21 %",
     "RESULTS 48a"),
    (r"\bdeployment\b(?!.{0,40}(setting is not|is not called))",
     "the matched-encoder condition is an UPPER BOUND, never 'deployment'",
     "THREAT_MODEL 3"),
    (r"held-out \(realistic\)|\brealistic\b(?=.{0,30}held-out)",
     "'realistic' is withdrawn -- nothing measures which condition deployment "
     "resembles", "council v4 B1"),
    (r"robust to purification|survives purification(?!.{0,30}stage 1)",
     "say what survives what, with the stage named", "FRAME_v6 4"),
    (r"min over purifiers|minimum over purifiers",
     "quantifies over an unbounded set; say 'an eight-configuration, "
     "three-architecture measured set'", "THREAT_MODEL 6a"),
    # This one gets its OWN exemption list. The global SAFE below is about
    # "is this line quoting the forbidden phrase in order to forbid it", and
    # a sentence that pairs the two encoders is not excused by containing the
    # word "fails" somewhere else in the same paragraph -- which is exactly
    # how the first version of this lint lost a real hit in THREAT_MODEL 6.
    (r"WavLM[- ]SV (and|&) ECAPA|ECAPA (and|&) WavLM",
     "never pair them as EQUAL evidence -- WavLM-SV sits at minDCF 0.77-0.91",
     "council v4 A.3",
     r"indistinguishable|two-group|never pair|not this|minDCF|embedding probe"
     r"|weakest|equal evidence"),
    (r"\bpredicted\b(?=.{0,40}(alpha|α|C1))|retro-validat",
     "C1 is an explanation generated AFTER observing the reversal",
     "FRAME_v6 4"),
    (r"criterion for choosing (ensemble )?members",
     "46 killed this; do not reintroduce it", "council v4 A.4"),
    (r"\bMOS\b(?!.{0,30}(predictor|UTMOS|NISQA))",
     "UTMOS/NISQA are MOS PREDICTORS, not opinion scores", "FRAME_v6 4"),
    (r"\b12 of 12\b.{0,40}p\s*=|\b8 of 8\b.{0,40}p\s*=|r\s*=\s*0?\.96.{0,20}p\s*=",
     "across-cell sign p-values and the r p-value are WITHDRAWN",
     "RESULTS 48c"),
    (r"four-way ranking|WavLM[- ]SV\s*>\s*ECAPA\s*>\s*x-vector\s*>\s*d-vector",
     "44: never a four-way ranking; x-vector and d-vector never separate",
     "RESULTS 44"),
    (r"the interaction vanishes",
     "write 'the matched-sized interaction is absent' -- the cohort could not "
     "detect one half that size", "RESULTS 52g"),
    (r"\bprotection method\b|\bwe propose\b",
     "R4 would not sign a 'protection paper'; this is measurement and limits",
     "council v4 A.5"),
]

# Lines that legitimately contain a forbidden phrase: ledger rows naming what
# NOT to write, and the appendix that lists them.
SAFE = re.compile(
    r"^\s*\|.*\|\s*$"                 # any markdown table row (ledger/appendix)
    r"|not this|forbidden|withdrawn|must not|\bNOT\b"
    r"|never[\s:*\"']|\bnever\b"       # "never write", 'never "x"', "Never:"
    r"|(do|does|did) not|cannot|neither|nothing (here |in )?\w*\s*measures"
    r"|rather than|instead of|no longer|killed it|is wrong|would have"
    r"|we are not|fail(s|ed)?\b|is absent|names? the set|supersede",
    re.IGNORECASE)


def main():
    # The draft is UTF-8 and this console is cp1252, so printing a line with
    # alpha in it killed the whole run. Replace rather than die: a lint that
    # stops at the first non-ASCII line is worse than useless, because it
    # reports "clean" for everything it never reached.
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    paths = sys.argv[1:] or DEFAULT
    total, flagged = 0, 0
    for p in paths:
        if not os.path.isfile(p):
            print(f"  skip (missing): {p}")
            continue
        print(f"\n=== {os.path.relpath(p, PROJ)} ===")
        lines = open(p, encoding="utf-8").read().splitlines()

        # Match on PARAGRAPHS, not lines. The prose is hard-wrapped at ~78
        # columns, so a rule's "unless the sentence also says X" lookahead ran
        # off the end of the line and fired on correct text -- six of the
        # first eight hits were that, and a lint with a 75 % false-positive
        # rate gets switched off. A markdown table row is its own paragraph
        # so ledger rows still match SAFE on their own.
        paras, buf, start = [], [], 1
        for i, line in enumerate(lines, 1):
            if line.strip() == "" or line.lstrip().startswith("|"):
                if buf:
                    paras.append((start, " ".join(buf)))
                    buf = []
                if line.lstrip().startswith("|"):
                    paras.append((i, line))
                continue
            if not buf:
                start = i
            buf.append(line.strip())
        if buf:
            paras.append((start, " ".join(buf)))

        hits = 0
        for i, para in paras:
            for rule in RULES:
                pat, why, where = rule[0], rule[1], rule[2]
                own = rule[3] if len(rule) > 3 else None
                if re.search(pat, para, re.IGNORECASE):
                    total += 1
                    # A rule with its own exemption list uses ONLY that list:
                    # the generic "this paragraph is quoting the phrase in
                    # order to forbid it" heuristic does not apply to it.
                    safe = (bool(re.search(own, para, re.IGNORECASE)) if own
                            else bool(SAFE.search(para)))
                    if not safe:
                        flagged += 1
                    tag = "ok(context)" if safe else "FLAG"
                    hits += 1
                    print(f"  [{tag}] line {i}  ({where})")
                    print(f"      {para.strip()[:170]}")
                    print(f"      -> {why}")
        if hits == 0:
            print("  clean")
    print(f"\n{total} matches, {flagged} outside a known-safe context.")
    if flagged:
        print("Each FLAG is a line to read, not necessarily an error.")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
