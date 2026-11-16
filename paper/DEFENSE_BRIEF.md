# Design brief — the defence paper

Written 2026-11-12. This is the plan for turning the project into a **defence**
paper. It sits on top of `PLAN_V4.md` (dates, budget) and `NEXT_STEPS.md`
(scientific content); where it differs, the difference is stated.

---

# PART I — THE CHOSEN PAPER: the narrow defence (decided 2026-11-12)

The user chose this over the hybrid. Part II below (the hybrid) is kept as the
broader option that was **not** chosen; nothing in it is scheduled.

## I.1 Topic, title and claims — FINALISED 2026-11-16

Supersedes every earlier version of this section. The 2026-11-12 drafts are in
git history; they carried three statements that later work falsified (that
segment crafting makes plain spend its budget, §63b; §59c's effect size, §64;
and a stale "cloner = DiffVC" scope line) and must not be quoted.

### Topic

Additive voice protection crafted with a diffusion purifier **inside** the
optimisation loop: how far it carries, and what the purifier-in-the-loop step
itself is responsible for.

### Title (working)

> **Voice Protection That Survives Diffusion Purification**

Fallback if a more explicit line is wanted: *Purification-Robust Speech
Perturbations Against Diffusion-Based Voice Cloning.*

The title leads with **C1**, the defence, because that is what this project set
out to build and because C1 is its strongest evidence. The held-out-encoder
limit (C3) goes in the abstract, not the title. The word "speaker" may be used
for the **encoders**; Control B's restriction (§54) applies to the alpha
instrument, which this paper no longer uses.

**The one-line statement of the idea, fixed 2026-11-16 and not to drift again:**

> An AntiPure for voice — a protective perturbation that survives diffusion
> purification and stops at least some diffusion-based voice cloning.

### Method name — DEFERRED (2026-11-16)

Not decided; decide before the draft's abstract is written, not after.

**One hard constraint whatever is chosen: the word "nullspace" must not appear
in the paper, and neither must the repository name.** `nullspace_voice_protection`
names the programme §54–§58 closed — there is no null space in this paper and
no α number survives. The folder keeps its name (renaming it mid-project would
break paths across the scripts, the run logs and RESULTS_ICASSP.md for no
scientific gain), but the paper must not inherit it.

Candidates discussed, with the objection to each recorded so it is not
re-litigated from scratch:

| name | for | against |
|---|---|---|
| **Indelible** | names the property C1 demonstrates — a mark that cannot be washed out; sits beside AntiFake / AntiPure / ClearMask without being derivative | mildly overclaims against C3 (72.4 % → 72.4 % held out), as every name in this field does |
| **PILOT** (Purifier-In-the-LOop proTection) | names the mechanism exactly, which is C2 | fits C2, and C1 leads; the acronym is heavily used elsewhere |
| **StillMine** | captures both halves — the listener still hears you, the cloner does not | unconventional for the venue |
| **Phantom** | already the project's informal name (the workspace folder) | names the defence after the *attack* — the phantom is the clone; and the reading that justifies it, "imperceptible mark", is the paper's **weakest** axis (PESQ 2.07, ~1 MOS below original, listeners 4.38 vs 5.00). Heavy collision risk in security (Phantom Cyber / Splunk) |

**A workable split if no single name wins:** keep **Phantom** as the project
and repository name, and give the *method* its own name in the paper. Projects
and methods routinely differ and nothing is lost.

Whatever is chosen, run the novelty check on the name before the abstract is
written (the standing rule in [[research-novelty-and-citations-workflow]]), and
if the name leans on imperceptibility, state the quality cost in the abstract's
first lines so the name is not promising what the paper then walks back.

### C1 — the defence holds, across purifiers and across cloners

A perturbation crafted with the purifier in the loop, at a psychoacoustic
budget, defeats **two** diffusion-based cloners and survives **two** diffusion
purifiers plus a stochastic attacker.

| | |
|---|---|
| DiffVC, through DiffPure v7 | ECAPA clone success **48.3 % → 0.0 %**; 9 of 9 cells, p_holm ≤ 2.9e-05 (§61a) |
| F5-TTS, through DiffPure v7 | ECAPA **86.2 % → 10.3 %**; 9 of 9 cells, p_holm ≤ 3e-04 (§60b) |
| a purifier never crafted against | 89–128 % of matched protection retained (§33) |
| stochastic attacker, best of 5 draws | WavLM 90–93 %, ECAPA 90–97 % protection (§28c) |

F5-TTS matters disproportionately: it is a 2024 flow-matching zero-shot model
with **no speaker-encoder module at all**, and it clones unprotected speech
better than DiffVC does (93.1 % vs 44.8–48.3 % on ECAPA). The defence was
never built against it and still holds.

**And it is not clone damage (§65).** Protection costs the F5-TTS clone
0.17–0.32 UTMOS and leaves WER untouched, while moving clone success from
86.2 % to 10.3 %. The protected clone scores 3.72 UTMOS — good speech that is
not the victim. Measured on the same instrument as §40c's DiffVC figures.

**The precise scope of that, because a reviewer will ask.** What is untargeted
here is the **cloner**, not the **verifier**: `fullnsT_tc_m30_seg` was crafted
with `--craft-encoders all`, so all three encoders that score the F5-TTS
clones were in the ensemble. C1 therefore says the damage survives a change of
*generative path*. It does **not** say the damage is encoder-general — C3 says
the opposite, and measures it. Never write "transfers to an untargeted model"
without saying which model.

### C2 — the purifier-in-the-loop step buys coverage, not strength

This is the paper's methodological contribution and the only claim that needs
a matched-budget comparison.

> At equal perturbation energy, putting the purifier in the crafting loop helps
> against an attacker that was **not** targeted, and does not help against the
> one that was.

Two independent budget-matched routes, which is why it is stated at all:

| route | what varies | result |
|---|---|---|
| §60d, §61c | the **cloner** (arms matched at 16.206 vs 16.113 dB, p = 0.408, purifier-aware at *better* PESQ) | F5-TTS **8 of 9** cells favour it, 5 significant; DiffVC **2 of 9**, and its two significant cells go **against** it, on the d-vector |
| §64 | the **encoder ensemble** (energy matched exactly, 0.000 dB) | **8 of 9** cells favour it, 1 significant; ECAPA −0.0354 [−0.0711, +0.0063] |

The cloner-varying route is the stronger and leads. §64 corroborates with its
interval and its power limitation shown; it is an **underpowered
not-significant, never a null** (§48's distinction, §52's n = speakers limit).

**And it is not bought with perceptual cost (§66a).** At the same perturbation
energy the purifier-aware arm has **better** perceived quality than plain:
UTMOS +0.293 [+0.232, +0.353], 28 of 29 speakers, p = 1.2e-09, with PESQ
(+0.192, §60d) and NISQA agreeing in sign. The one-line form: *equal energy,
better quality, and the advantage is on the untargeted attacker.*

Two things travel with C2 wherever it appears:
* **Name the attacker.** On DiffVC's own conditioner the effect reverses, and
  §62 shows both components cause that — EOT alone costs d-vector protection
  (+0.0346, p 0.0059) and the noise-matched regulariser costs as much again
  (+0.0246).
* **The regulariser is an ablation that does nothing**, reported as such, never
  as a component of the method.

### C3 — the limit, measured with the same protocol

Protection does **not** transfer to a speaker encoder held out of the crafting
ensemble, and the failure is total in the worst cell:

| held-out encoder (d-vector), channel | clone success |
|---|---|
| no purifier | 93.1 % → 69.0 % |
| DiffPure v7 | 82.8 % → 62.1 % |
| **DiffWave (foreign purifier)** | **72.4 % → 72.4 %** (p = 0.47, no protection) |

A held-out encoder behind a purifier we never crafted against gets nothing.
A phoneme-guided refiner (full PhonePuRe) reduces protection further, to
38 / 14 / 59 % clone success unfiltered (§29e).

C3 is not an afterthought: it is why C2 is interesting. The component that
helps is exactly the one that helps against what you did not target, and it is
still not enough to cover a held-out encoder.

### What this paper is NOT, and will not claim

* **Not a null-space or subspace-geometry paper.** §54–§58 closed that: the α
  contrast is reproduced by a random projection of the front end with no
  network in it. No α number appears.
* **Not the on-manifold / formant-warp hybrid** (Part II below). Not chosen,
  not run.
* **Not a measurement-only paper.** The measurement findings (§22b, §29, §40c,
  §42h) are motivation and limits, not the lead.
* **Not "purifier-aware crafting protects better"** unqualified — false in the
  matched condition (§59b, §61b).

### Fixed conditions, stated wherever a number is

Purifiers: mel DiffPure v7 (DDIM, t = 200) and DiffWave (t = 3). Cloners:
DiffVC and F5-TTS v1 Base. 29 cross-session LibriSpeech test-clean speakers,
4-s crops, enrolment from a different chapter, cross-session EER thresholds
(WavLM-SV 0.8926, ECAPA 0.3358, d-vector 0.7029), ECAPA additionally at the
0.25 field convention. Budget m30_f15. **Every cloning comparison comes from a
single run** (§62b) and new runs pass `--clone-seed`.

## I.2 What supports it, already measured

| evidence | source | number |
|---|---|---|
| clone success on crafted-against encoders | §22b | 12–14 % (vs 75–92 % held out) |
| paired clone margin, unfiltered, n = 29 | W1 2a | ECAPA −0.182, WavLM −0.102, d-vec −0.098, p ≤ 1.1e-8 |
| stochastic attacker, best of 5 draws | §28c | WavLM 90–93 %, ECAPA 90–97 %, d-vec 62–83 % protection |
| transfer to a purifier not crafted against | §33 | 89–128 % retention |
| seed stability of continuous quantities | §31 | spread ≤ 0.009 |
| quality of the shipping arm | §40 | on disk |

Every number above must be re-read from its CSV before it enters the draft
(`check_claims.py`), and the binary rates quoted with seeds pooled (§31 rule).

## I.3 What is missing

| # | item | GPU-h | why it is needed |
|---|---|---|---|
| ~~N1~~ | **DONE 2026-11-12, zero GPU — RESULTS §59.** The contrast was already on disk in `cloning_tc_m30_deploy.csv` and `cloning_tc_m30_nodvec.csv`, never analysed. It changed the claim (§I.1) and killed the regulariser (§59d). | 0 | — |
| ~~N2~~ | **DONE 2026-11-12 — RESULTS §60. OUTCOME N2-A: it transfers.** F5-TTS (encoder-free, flow matching), 9 of 9 cells Holm-significant; ECAPA clone success 86.2 % → 10.3 % through the purifier. The pre-registered prediction (60 % no transfer) was wrong and is recorded as such. | 0.5 | — |
| N3 | **Listening data**, even 10–15 listeners, on the shipping arm | 0 | "still sounds like the same person, acceptable quality" |
| ~~N4~~ | **DROPPED 2026-11-16 — RESULTS §66c.** The figures C3 quotes (§29e clone success) are scored at the clean calibration *by design* and are not provisional. Only §29b's ASV acceptance is, and it cannot be calibrated well: PhonePuRe outputs exist for 29 reference files (one per speaker) against the clean calibration's 464 target trials. | 0 | — |
| ~~N8~~ | **DONE 2026-11-16 — RESULTS §65.** Protection costs the F5-TTS clone 0.17–0.32 UTMOS and 0.35–0.75 NISQA (6/6 cells significant) and **zero WER** (all intervals span 0). Small beside 86.2 %→10.3 % clone success, and the protected clone at 3.72 UTMOS is still good speech — it just is not the victim. Matches §40c's DiffVC figures on the same instrument. | 0.5 | — |
| ~~N5~~ | **DONE 2026-11-12 — RESULTS §61. Outcome 5-C on the pre-registered primary** (which was saturated, my error). At matched budget through DiffVC, purifier-awareness does not help and on the d-vector it hurts; through F5-TTS it does help. Claim 2 above is the result. | 1 | — |
| ~~N7~~ | **DONE 2026-11-12 — RESULTS §63. OUTCOME 7-D: the budget match FAILED** (plain landed at 19.001 dB, 2.878 dB adrift, p = 2.6e-11), so no verdict. It did falsify §60d: plain's spend is set by the crafting **ensemble**, not by segment crafting, and a matched held-out arm **cannot be crafted at the same nominal budget**. | 2.7 | — |
| ~~N7a~~ | **DONE 2026-11-12 — RESULTS §64. OUTCOME 5-B.** Energy matched exactly (0.000 dB). 8 of 9 cells favour purifier-awareness, 1 Holm-significant; ECAPA −0.0354 [−0.0711, +0.0063]. Partly budget, partly method — underpowered, not null. Feeds C2's second route. | 2.3 | — |
| ~~N6~~ | **DONE 2026-11-16 — RESULTS §66a**, free as part of the §65 run. **The purifier-aware arm is BETTER quality than plain at the same energy: UTMOS +0.293, 28/29 speakers, p = 1.2e-09.** Both arms cost ~1.1–1.4 UTMOS against the original with WER untouched. | 0 | — |

**N2 design.** Cloner: F5-TTS (flow-matching zero-shot TTS, no separate
speaker encoder — it reads the voice from the audio prompt). Arms: clean
reference, `plain_m30_f15`, shipping recipe; channels: none, purifier v7,
DiffWave. Score with the same three encoders and thresholds; headline is the
paired continuous margin, n = 29. Installed in its own venv, like
`external/phonepure_venv`. Pre-register the endpoint before running.

**N2 can go against us.** The perturbation is built against speaker encoders
and F5-TTS has none, so protection may not carry over. Both outcomes are
written in advance:
- holds → the abstract may say "and transfers to a modern zero-shot cloner";
- fails → it is the third stated limit, and the title keeps "DiffVC"-scoped
  wording. It does not sink the narrow claim.

## I.4 Limits the paper states itself

1. Cloner's encoder unknown → cloning succeeds 75–92 % (§22b).
2. Phoneme-guided refiner (full PhonePuRe) → 38 / 14 / 59 % unfiltered (§29e).
3. ~~Whatever N2 returns~~ — N2 came back positive (§60), so this is no longer
   a limit. What remains from it: **no quality measurement on the F5-TTS
   clones**, so §40c's "part of protection is clone damage" is untested for
   this cloner; and one modern cloner is not a class.
4. One corpus, 29 speakers.
5. Two of the three pre-registered predictions this project has scored were
   wrong (§7.5, §60c). Say so.

## I.5 The novelty risk, and the answer to it

**The risk, named precisely (2026-11-16).** The method exists in the image
domain, in three layers, and a reviewer who knows that literature will say so:

| layer | prior art | what it already does |
|---|---|---|
| the purifier | **DiffPure**, Nie et al., ICML 2022 | diffusion purification as an adversarial defence — invented for images |
| the crafting technique | **DiffAttack** (NeurIPS 2023, arXiv 2311.16124), **DiffHammer** (NeurIPS 2024), DiffPure's own adjoint method | optimising a perturbation *through* the diffusion purification loop — technically what we do |
| **the whole application** | **AntiPure**, Yang et al., **ICCV 2025**, arXiv 2509.13922 | *"we formalize the anti-purification task"* — purification-resistant **protective** perturbation, same role structure as ours (perturbation = defence, purifier = attacker's tool), in images, for diffusion customization |

`related_work.md` §3b calls AntiPure "the closest prior art to our
purifier-aware crafting" and §10.5 says: **expect "isn't this AntiPure for
audio?" and answer it pre-emptively.** Treat that as mandatory.

**C1 still leads, and that is a considered choice (revised 2026-11-16).** An
earlier draft of this section said C1 was "a weak lead" and moved the weight
onto C2/C3. That over-corrected, for two reasons:

1. **Domain transfer is a normal Interspeech contribution.** It is a speech
   conference; speech-specific design is the point, and an image-domain
   precedent costs far less here than at a vision or general-ML venue. The
   lineage is conceded in one sentence and the paper moves on.
2. **C1's evidence is strong and C2's is thin.** C1 is 9 of 9 cells on two
   cloners and two purifiers plus a stochastic attacker; C2's energy-matched
   test yielded one significant cell of nine. Leading with the strongest
   result and a known precedent beats leading with a novel thin one — and C2
   is additionally a partly *negative* finding about our own method, which is
   excellent as supporting analysis and uncomfortable as a headline.

**The answer, in two parts.**

*1. The audio gap is real and specific.* Every existing audio protection claims
robustness to the **wrong** threat: SafeSpeech (USENIX Sec 2025) and POP to
noise reduction and augmentation, RoVo (2505.12686) to speech enhancement,
ClearMask (ASIA CCS 2025) to signal recovery, AntiFake (CCS 2023) to neither.
**None claims robustness to diffusion purification** — while De-AntiFake
(ICML 2025) shows the diffusion attacker is already here. AntiPure is images
and does not touch speech encoders, thresholds, or intelligibility.

*2. C2 and C3 are the credibility, and nobody has done them in either domain.*
They are not the headline; they are what makes the headline believable. No image
paper and no audio paper has asked, at **matched perturbation energy**, what
the purifier-in-the-loop step is actually responsible for — and answered that
it buys transfer across *cloners* and not across *encoders*. AntiPure proposes
a method and shows it works; it does not decompose it against a plain baseline
at equal budget, across two generative models, with the failure case measured.
That decomposition is the contribution, and the title already leads with it.

Carried over, still ours: protection scored **end-to-end on the clone** rather
than on ASV acceptance alone; a **stochastic best-of-5** attacker rather than
the single draw crafted through; **transfer across purifiers** never seen; and
limits measured with the same protocol rather than omitted.

**Consequence for the writing.** The introduction must cite AntiPure in its
**first paragraph of related work** and concede the method lineage in one
sentence — explicitly, not by omission. Then C1 leads on the speech gap no
audio protection has claimed, and C2/C3 follow as the analysis that earns
trust. A paper that buries AntiPure in a citation list invites the one review
we cannot answer; a paper that concedes it in a sentence and then shows two
cloners, a matched-budget decomposition and its own failure case does not.

Forbidden wordings carried over: "reverses De-AntiFake" (stage 1 only),
"equivalent" for BPDA vs exact (§51), "replicated 3x" (§30e). Added: never
present purifier-in-the-loop crafting as novel in itself.

## I.6 Paper outline (4 + 1 pages) — tracks C1/C2/C3 of §I.1

1. **Introduction.** Diffusion purification is the attacker's countermeasure
   and no audio protection claims robustness to it, while De-AntiFake shows
   the diffusion attacker has arrived. AntiPure (ICCV 2025) did this for
   images; the lineage is conceded here, in one sentence, in the first
   related-work paragraph.
2. **Threat model.** The attacker's chain; matched vs held-out defined before
   any number is shown, because the whole paper turns on that distinction.
3. **Method.** Purifier-in-the-loop EOT, ensemble loss, budget. The
   noise-matched regulariser is reported as an **ablation that does nothing**
   (§59d), not as a component.
4. **Results.**
   (a) the defence holds through both purifiers, 9/9 cells (§59a);
   (b) **the attribution** at matched budget (§61c, §64) — the analysis that
       earns C1, not the headline;
   (c) EOT is the active ingredient, the regulariser is not (§59d);
   (d) the stochastic best-of-5 attacker (§28c);
   (e) transfer to unseen purifiers (§33);
   (f) the modern encoder-free cloner, F5-TTS (§60) — outcome N2-A;
   (g) quality, and listeners if N3 delivers.
5. **Limits — I.4,** led by the full PhonePuRe refiner.

**Figures, both built.**

* **Fig. 1 — `paper/fig_n1_coverage.png`** (`make_fig_n1.py`): clone success for
  no-protection / plain / purifier-aware, encoder-in-ensemble beside
  encoder-held-out, DiffVC. Saturated cells hatched. Shows the null and the
  effect in one picture.
* **Fig. 2 — `paper/fig_cloners.png`** (`make_fig_cloners.py`): the §61c result.
  Paired delta with BCa intervals, DiffVC beside F5-TTS, same arms and budget;
  the noise floor shaded, saturated cells hollow. The left panel sits above
  zero and the right below it, which is the paper's whole argument in one
  image.

Fig. 2 is the stronger of the two and should lead if only one fits.

## I.7 Calendar

| by | what |
|---|---|
| 11-19 | N2 pre-registered, cloner installed and smoke-tested on 2 speakers; listening test live |
| 11-30 | N1 and N2 reported in `RESULTS_ICASSP.md` |
| 12-10 | N4 done; all tables regenerated on the current base |
| 12-24 | first full draft |
| 01-15 | compute stop; listening data closed |
| 01-29 | review pass, `check_claims.py` green |
| 02-09 | submit to Interspeech 2027 |

GPU total 5–13 h, well inside budget. The binding risk is N3 (recruiting
listeners), not compute.

---

# PART II — NOT CHOSEN: the hybrid defence

## 1. The claim the paper must earn

> A voice protection that (a) still works on a speaker model it was **not**
> built against, (b) still works after a **restoration** model, and (c) is
> still heard as the **same person**.

The additive method fails (a) and (b). Nothing published is tested on all
three. The paper is a defence paper only if all three are measured and at least
(a) and (b) beat the additive baseline at a quality cost listeners accept.

## 2. What the current method gets wrong

| failure | measured | cause |
|---|---|---|
| no transfer | clone success 12–14 % targeted vs 75–92 % untargeted (§22b) | gradient of one encoder does not carry to another |
| restoration undoes it | 38 / 27 / 71 % after PhonePuRe (§29) | the protection is added noise, so it can be subtracted |

Both are structural. More encoders or a different loss will not remove them.

## 3. The method: a two-part hybrid

**Part A — speaker-adaptive formant transform (carries transfer and survival).**
- Start from `t_mcadams` in `utils/onmanifold.py`.
- Replace the single global α with a few parameters per speaker: a warp
  strength per formant band, chosen for that speaker.
- Search is low-dimensional (≤ 8 parameters): grid or CMA-ES, not PGD.
- Objective: lower the speaker score on an encoder ensemble **after** the
  purifier, under a hard naturalness limit.

**Part B — small purifier-aware additive residual (adds strength).**
- The existing recipe (`exp_purifier_aware.py`, BPDA, seg crafting), applied to
  the output of Part A at a tighter budget than m30_f15.

**Why together.** Each part runs at a lower dose than it would alone. Part A
gives what a restorer cannot subtract; Part B gives margin on the targeted
models without pushing the warp into the "different person" range.

**Naturalness limit.** Set by the listening test (Section 5), not by a MOS
predictor. Until listener data exists, use UTMOS ≥ 3.0 as a placeholder only.

## 4. Experiments, in order, with stop rules

| # | experiment | GPU-h | decides |
|---|---|---|---|
| E1 | Fixed warp on the current base: untargeted encoder + PhonePuRe (already pre-registered, `PREREG_step1_warp_remeasure.md`) | 12–16 | Is Part A worth building? |
| E2 | Listening test: same/different person across warp strengths (`PREREG_step0_listening_frontier.md`) | 0 | The naturalness limit |
| E3 | Per-speaker optimised warp, pilot on 3 dev-clean speakers | 4–6 | Does adapting per speaker beat a global α at equal quality? |
| E4 | Hybrid A+B, pilot on 3 dev-clean speakers | 3–4 | Does B add anything on top of A? |
| E5 | Full run, 29 test-clean speakers, all arms | 15–20 | The paper's main table |
| E6 | Second cloner with a different speaker encoder, evaluation only | 4–6 | "Not just DiffVC" |

**Stop rules.**
- E1: if warp does not beat the additive method on the untargeted encoder, or
  collapses under PhonePuRe → stop the defence paper, go to the measurement
  paper.
- E2: if no warp strength is heard as the same person → same.
- E3: if per-speaker warp does not beat global α on the pilot → drop
  adaptation, keep global warp + residual as the method (weaker novelty).
- E4: if the residual adds nothing → the method is Part A alone.

Pilots run on dev-clean only, never on the 29 confirmatory speakers.

## 5. The main table (what a reviewer will look for)

Rows: no protection · additive (current recipe) · global warp · adaptive warp ·
hybrid.

Columns:
1. clone success on **targeted** encoders
2. clone success on an **untargeted** encoder
3. clone success after **purifier v7**
4. clone success after **PhonePuRe**
5. quality: UTMOS, WER
6. listeners: % "same person"

The paper stands if the hybrid row wins columns 2 and 4 and holds column 6.

## 6. Novelty statement

- ClearMask: on-manifold transforms, never tested against any purifier.
- De-AntiFake, VocalBridge: attack additive protections only.
- This paper: first protection **optimised with the restorer in the loop and
  scored on untargeted models**, with a per-speaker transform.

The measurement findings (§22b, §29, §40c, §42h) become Section 2 of the paper:
the motivation, and the evaluation protocol the defence is held to.

## 7. Paper outline (4 + 1 pages)

1. Introduction — the three requirements, and that current protections meet one.
2. Why additive protection fails — coverage gap and restoration, two numbers.
3. Method — adaptive transform, residual, joint objective.
4. Setup — 29 speakers, encoders, purifiers, cloners, listening test.
5. Results — the main table, then the quality/protection frontier figure.
6. Limits — what still breaks it.

## 8. Calendar

| by | what |
|---|---|
| 11-26 | E1 reported; listening test live and collecting |
| 12-10 | E2 reported; E3 and E4 pilots reported; method frozen |
| 01-08 | E5 and E6 complete |
| 01-15 | compute stop |
| 02-09 | submit to Interspeech 2027 |

This is two weeks behind `PLAN_V4` and uses its one slack week. Total 38–52
GPU-h against roughly 40–50 available, so there is no room for re-runs.

**If the method freezes late or E5 slips:** submit the measurement paper to
Interspeech and send the defence paper to ASRU 2027 (~May) or ICASSP 2028,
rather than submitting a defence with a missing column.

## 9. Open risks

- Listening test has zero responses; recruitment is unsolved and it gates the
  whole paper.
- The warp numbers that motivate this come from the superseded base (20
  speakers) and may not reproduce.
- Only one cloner is wired up today; E6 needs a second one chosen and ported.
- The mechanism that makes warping transfer (it changes what every speaker
  model reads) is the same reason listeners may notice it.
