# Reviewer council v3 — 2026-10-04

Five independent reviewers read RESULTS §1-32, the drafts, plans, code and CSVs
(not just prose). Target: Interspeech 2027 (9 Feb 2027).

| Reviewer | Score | Vote |
|---|---|---|
| R1 adversarial ML | 4/10 | reject as framed; 6-7 accept if reframed + controls #3, #10 |
| R2 speech / ASV | 4/10 | reject as evidenced; accept path via disjoint calibration + test-other + quality |
| R3 statistics | trust 6.5/10 (was 7) | will not sign off claims as written |
| R4 security | 4/10 | reject as framed; 6/10 weak accept if reframed |
| AC | Interspeech ~35 % (reframed) / 10-15 % (as defence); SPSC ~70 % | pick framing (b): analysis paper |

Why lower than council v2 (~6): this time the reviewers opened code and CSVs and
found defects the notebook does not know about (#2, #3, #4, #5, #8 below).

**Unanimous:** stop calling it a protection/defence paper; write an analysis
paper led by §32 (encoder geometry), with §29 (PhonePuRe) and §30 (BPDA = exact)
as headline findings. **Do NOT build the reduced-N adaptive attack through
PhonePuRe stage 2** (R1: only if #3 and #10 come back favourable, start by 1 Nov,
hard abort date).

Verified by hand on 2026-10-04: `foreign_tc_*.csv` + `survival_tc_v7.csv` exist
(Sep 20) with no RESULTS section; pre-filter at `exp_cloning_protection.py:307`
still unions across purifiers; `--n-eot` default 1; no minDCF anywhere.

---

## A. Ranked weaknesses — FIXABLE / PARTIAL / NOT (before 9 Feb, one RTX 3050)

| # | Weakness | Who | Fix | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | **Thesis is dead as written.** Both novelty claims gone (§29 PhonePuRe recovers speaker; §30 estimator not the delta). Draft still sells a protection. | all 5 | Reframe as analysis/scoping paper: (1) §32 encoder geometry + criterion, (2) the constraint behaviour it explains, (3) measured ceiling: §22b encoder coverage, §30 BPDA null, §29 PhonePuRe. Retire PLAN_TO_CLOSE_THE_CHAIN (it serves the dead defence paper). | 0 GPU, ~2 wk writing | FIXABLE |
| 2 | **Two evaluation bases mixed.** Lead result quoted on old base (86-107 %, n=20, p=1.9e-06); current-base CSVs on disk unanalysed (89/128/91/126 %; survival α 0.879 vs 0.743 = 1.18x, not draft's 0.97 vs 0.49). TABLES Table 1 still ECAPA EER 0.03 %. `make_paper_tables.py:213`, `make_latex_tables.py:119` read `*_n20.csv`. | AC, R1, R3, R4 | Write §33 from `foreign_tc_*` / `survival_tc_v7`; regenerate every table; nothing before §21 appears as a number. | 0 GPU, 1-2 days | FIXABLE — do first |
| 3 | **Mechanism control missing.** Random baseline = isotropic white noise (`exp_nullspace_survival.py:200`, `exp_encoder_subspace.py`) vs a denoiser trained to remove white noise. Vocoder alone already gives 1.36x (§11b). "Subspace" may be "smooth/band-limited survives". | R1 | Add spectrally-matched random (same per-band power) and clean-PCA-span random; report next to every α. If optimised still wins → spine holds; if not → honest negative result. | 3 h code, 1.5 GPU-h | FIXABLE — could flip the story, run before writing |
| 4 | **Cloning pre-filter selects on a noisy outcome.** 56/174 (32 %) unprotected cells flip between runs (no perturbation involved); keep-set n varies 9/6/7; biased toward the defender (winner's curse). Union-across-purifiers bug still live in `exp_cloning_protection.py:307`; correct version only in `analyse_cloning.py`. | R3, R2, R1 | One-line groupby fix / import `analyse_cloning.prefilter`. Gate on R=5 unprotected draws. Headline = paired continuous margin on all 29, unfiltered; rates only on fixed intersected keep-set with Clopper-Pearson + exact McNemar. | 0 GPU re-analysis + 6-10 GPU-h gate | FIXABLE |
| 5 | **No out-of-sample validation anywhere.** Thresholds calibrated on the same 29 speakers they score (464 = 29x4x4 targets). Mechanism claim (C2) recovered after a failed confirmation (§25 budget switch); Holm family redrawn 3x. `DEANTIFAKE_SPEAKERS` and dev-clean unused. | R3 #1 ask, R2 #1 ask | Pre-register 12 cells in writing, then re-run only `full` + `fullnsT` with BPDA on dev-clean (or test-other); recalibrate thresholds + 4 foreign channels on a disjoint cohort. Turns C2 from hypothesis into finding. | ~1 day code, ~8 GPU-h | FIXABLE — single highest-value experiment |
| 6 | **No quality evidence on the shipped arm.** `quality_nr.csv` is dev-clean, non-seg; shipping arm `fullnsT_tc_m30_seg` has only PESQ 2.2 / SNR 15 dB. No NISQA. SQUIM is intrusive-PESQ predictor mislabelled as naturalness. WER uses Whisper-small's own transcript as reference over ~244 words. Listening test: responses/ empty, stimuli are stale non-seg arms, condition name visible in filenames, no loudness normalisation. | R2, R4, R1, AC | Rerun `eval_quality_nr.py` on shipping arm (29 spk) + NISQA + real-transcript WER (MFA alignments already on disk from §29a, whisper-large-v3, EnglishTextNormalizer). Regenerate listening stimuli, -23 LUFS, blind filenames, 20-25 listeners (Prolific ~$150). | ~16 h code, ~2 GPU-h, 1-2 wk calendar | FIXABLE if started week 1 |
| 7 | **Threat model stale and self-contradictory.** THREAT_MODEL.md (Jul): defender knows purifier (contribution lives in mismatch); formal success needs PESQ > 3.0 (shipping 2.2 never meets it); protection params assumed secret (violates Kerckhoffs); still says "null space". "Deployment" = matched-encoder white-box on DiffVC's own conditioner. | R4, R1 | Rewrite: public defence, unknown purifier, encoder match as explicit axis; rename settings "matched-encoder (upper bound)" / "held-out (realistic)"; success = min over attacker options. | 0 GPU, ~1 day | FIXABLE |
| 8 | **Inferential wording.** Floor p-values quoted as strength (1.9e-06 = 2/2^20, 3.7e-09 = 2/2^29). "Retention" >100 % with no CI. §32: x-vector vs d-vector p = 0.54 / 0.47 unreported → ordering is two groups, not four; "retro-validation" is post-hoc. BPDA null needs TOST equivalence. §27b bootstrap code not committed (`analyse_mismatch_final.py` missing). Block-(a) control nulls uncorrected. | R3 | All re-analysis: sign-test framing; log-ratio + bootstrap CI for retention; all 10 pairs Holm / Friedman-Nemenyi; TOST |Δ|<0.01; mixed model `delta ~ purifier*t*encoder + (1|speaker)`; BCa; commit the script. R3 wrote replacement sentences for C1, C2, C3, C7. | 0 GPU, 3-4 days | FIXABLE |
| 9 | **Attacker compute parity.** EOT size 1 (`exp_purifier_aware.py:671`) while De-AntiFake/VocalBridge sweep 1/5/10/15. Protection scored on mean-over-seeds; §28 best-of-k confounded (3 DDIM vs 5 DDPM seeds). | R1, R3, R4 | Say "EOT size 1"; ablate EOT=5 on n=10; best-of-N as headline at matched k; store per-seed rows. | ~1 day, ~8 GPU-h | FIXABLE |
| 10 | **Front-end confound.** WavLM/ECAPA evaluated through the exact crafting function; only the d-vector has a different eval path (VAD + partials) — and it's the one that fails. | R1 | Re-score saved wavs through VAD-trim + normalise + ±2 % resample. If WavLM/ECAPA hold, confound dead. | 3 h, 1 GPU-h | FIXABLE |
| 11 | **Purifier v7 overclaimed.** Residual, time-unconditioned, denoise gain 0.19 (~81 % identity). Transfer = one architecture (DiffWave), two checkpoints; the one different architecture (NCSN++, §29) breaks it. | R1 | Disclose gain/conditioning per purifier; rename v7 "iterated mel denoiser in a DiffPure-style loop"; collapse Holm family to architectures. Optional: SGMSE+/StoRM as 3rd architecture. | prose; optional ~1 day + 6 GPU-h | PARTIAL (better purifier: NOT) |
| 12 | **Lossy channel untested** (MP3/Opus/AAC, loudness norm, 8 kHz). Only data is old-base 3 files, 33 % hold. δ band-limited at 6 kHz. ffmpeg missing. | R4 (first question in review), R2 | Install ffmpeg; codec suite, evaluation-only on saved wavs. | ~4 h code, 2 GPU-h | FIXABLE |
| 13 | **Dated encoder set.** WavLM-base-plus-sv, 2020 ECAPA, 2018 GE2E (= DiffVC's conditioner). No CAM++/ERes2Net/TitaNet/ReDimNet. §32 measured on mel purifier while reversal is on DiffWave. | R2, AC, R1 | (a) 2 modern cosine-scored encoders as eval-only (WeSpeaker / 3D-Speaker). (b) Prospective test: measure α(g_e) for CAM++ before crafting, commit prediction, then craft — converts §32 into a prediction. (c) §32 through DiffWave (waveform adapter). | ~25 GPU-h total | FIXABLE |
| 14 | **λ = 10 single value; "trade" has no cost term** on current base (§27c null unpurified; constrained arm quieter). | R1 | λ ∈ {1,3,10,30} at n=10; stop calling it a trade; put the d-vector reversal in the abstract. | ~5 GPU-h | FIXABLE |
| 15 | **McAdams baseline under a weaker attacker** — only ASV via v3; never through DiffWave / PhonePuRe / DiffVC. Its one win is the d-vector. | R2 | Render McAdams into `WAV_DIR`, run identical cells; LPC order 20. | 4 h, 2 GPU-h | FIXABLE |
| 16 | **One cloner (DiffVC).** Baselines use 5-6. | R2 (required), R4; AC says cut under analysis framing | OpenVoice V2 tone-colour converter (HF mirror; different conditioner → true strict number). | ~10 h code, 4 GPU-h | FIXABLE — do only after 1-10 |
| 17 | **Cheap unmodelled attacker variations** (multi-clip aggregation, window cropping, speech-enhancement preprocessing). | R4 | Evaluate and report; a self-reported break is a finding, a reviewer-found one is a reject. | ~10 h code, ~6 GPU-h | FIXABLE |
| 18 | **No directly comparable row** to AntiFake/De-AntiFake. `DEANTIFAKE_SPEAKERS` unused; `succeeds_at_025` computed, never reported. | R2 | One row on their speaker list + NISQA + their 0.25 convention; state protocol deltas in captions elsewhere. | 6 h, 3 GPU-h | PARTIAL |
| 19 | **ASV reporting conventions.** No minDCF, no DET, EER 0.62 % = ~3 errors with no CI; legacy `thresholds_clean.csv` is still the silent default in the cloning script. | R2 | EER + minDCF (p=0.01/0.05), speaker-bootstrap CIs, acceptance-vs-threshold curves / ΔEER; `load_thresholds` must fail on mismatched base. | 6 h, 1 GPU-h | FIXABLE |
| 20 | **Threshold leakage + best-iterate selection** on matched-purifier score (`--thr-aware`, `craft()` best_delta). Doesn't touch foreign cells. | R1 | Disclose; matched cells = sanity check; report margins. | prose | FIXABLE |
| 21 | **Defender cost unreported** (139-259 s GPU per 4-s clip). | R4 | One paragraph. | 1 h | FIXABLE |

## B. Cannot be solved before 9 Feb on this hardware — state as limitations

1. **Full-strength adaptive attack through PhonePuRe stage 2.** Backward peaks at 11.3 GB on a 6 GB card; only a reduced-N (N=6) version fits, which reviewers will discount. Needs a ≥24 GB GPU for ~2 weeks. *This is the single resource that separates this from a top-tier paper.*
2. **Scale.** n ≥ 100 speakers across ≥ 2 corpora; spontaneous/conversational, noisy, cross-lingual speech. test-other / dev-clean give ~50 speakers at most.
3. **Baseline parity on cloners.** 5-6 VC/TTS systems + commercial services is weeks of a second person.
4. **A strong, time-conditioned purifier.** Retraining invalidates every v7 number.
5. **Selection history of the mechanism claim** (§25 post-hoc budget rescue). Can only be disclosed, or superseded by #5's pre-registered replication.
6. **Paradigm-inherent (no experiment fixes these):** protection is encoder *coverage*, not generalisation (§22b: 12 % vs 88 %); published audio is permanent while purifiers improve (§29 is an instance); the attacker needs one clip, including any never-protected one; the harm is judged by humans, the objective by encoders; the user pays the quality cost with certainty while the benefit is probabilistic.

## C. Where reviewers disagree

- **Second cloner:** R2 says required; AC says cut under analysis framing. → do after items 1-10; mandatory if falling back to SPSC.
- **§32 strength:** AC/R2 call it the best asset; R3 shows the bottom of the ordering is tied (p = 0.54). → claim two groups (WavLM vs rest) + d-vector nsr > 1; make it predictive via the CAM++ prospective test (#13b).
- **Underclaimed (R1):** §30 BPDA ≡ exact at half the compute is a field-correcting null; §29's ASV-proxy vs end-to-end divergence (55/76/100 % acceptance vs 38/27/71 % clone success) is a methodological finding; §24b (x-vector flatters tables) is a generalisable warning.

## D. Plan (AC, merged with R1/R2/R3)

| Dates | Work | GPU |
|---|---|---|
| Oct 5-11 | #2 §33 + tables; #4 pre-filter fix; #8 re-analysis; #7 threat model rewrite. Start listening-test rebuild + recruitment (#6). | 0 |
| Oct 12-18 | #3 matched random control; #10 front-end re-score; NISQA + real-transcript WER. | ~5 h |
| Oct 19-Nov 8 | #5 pre-registered dev-clean replication with disjoint calibration; #12 codec suite. | ~12 h |
| Nov 9-29 | #13 CAM++ prospective test + §32 through DiffWave; #9 EOT=5 / best-of-N. | ~25 h |
| Nov 30-Dec 20 | Contingent: #14 λ sweep, #15 McAdams, #16 OpenVoice, #17. **GPU stop 20 Dec.** | ~15 h |
| Dec 21-Jan 18 | Listening test analysis; full draft from current base only. | 0 |
| Jan 19-Feb 9 | Red-team, figures, buffer. arXiv on submission day. | 0 |

Venue: Interspeech 2027 primary (framing b). SPSC Symposium fallback. If by
20 Jan there is no human data, submit to SPSC instead of a weak Interspeech paper.
