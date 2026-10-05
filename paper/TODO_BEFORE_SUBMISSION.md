# What still needs fixing before this paper is submittable

*Written 2026-09-16. Current draft: `paper/draft_v5.md`. Results of record:
`experiments/RESULTS_ICASSP.md`. Prior-art ledger: `paper/related_work.md`.*

Ordered by whether it blocks submission, not by effort. Each item says what is
wrong, why it matters, and what "done" looks like.

---

## A. BLOCKERS — the paper is not submittable until these are closed

### A1. The paper reports survival in two different metrics ⏳ IN PROGRESS
`draft_v5` §4.1 quotes PSR; the abstract quotes signal gain α. RESULTS §11a
establishes α as the correct one — a norm ratio cannot separate "survived" from
"destroyed and replaced by equal-sized noise", which is exactly what Griffin-Lim
does. Shipping both invites a reviewer to notice the inconsistency and distrust the
rest.

**Status:** rerun running (`nullspace_survival_20spk_u3_alpha.csv` done, u6 in
progress, probe after). **Done =** §4.1 table and the double-dissociation table in
§5.4 both in α, PSR retained only in the CSVs for continuity.

### A2. No cloning model has ever been run ⏳ IN PROGRESS — DiffVC integrated, see §B
The entire motivation is voice cloning; every number we have is speaker
verification. This is the single most likely reason for rejection, and unlike the
other gaps it cannot be argued away in a limitations paragraph. Plan in §B.

### A3. Citations ✅ DONE
All 13 verified against primary sources on 2026-09-16; six were wrong.
`paper/references.bib` now holds the verified entries with provenance comments.
Corrections were: DiffWave is ICLR 2021 **oral**; Kadkhodaie is oral **+
Outstanding Paper**; Shallow Diffuse is **NeurIPS 2025 Spotlight** (not a preprint)
and the title is "Low-**Dimensional**"; AntiFake's full title and pages 460–474;
"Liu Yuezhang" is surname-first so "Yuezhang & Wei" is wrong; Miyasawa has **no
issue number**; Tramèr et al. was never formally published.

*Process note:* openreview.net and dblp now sit behind bot challenges. Use the
arXiv export API, Crossref, PMLR and CVF Open Access instead.

---

## B. THE CLONING EVALUATION (the new work)

Decision: **two voice-conversion systems, DiffVC and OpenVoice V2.** Rationale from
the verified frequency table over six baseline papers — DiffVC is used by
De-AntiFake, VoiceCloak and ClearMask (3/7), OpenVoice by De-AntiFake (V2) and
SafeSpeech (V1) (2/7). Both are VC, which matches the realistic "clone my voice
from a posted clip" threat. `kNN-VC`, `FreeVC`, `SpeechT5` and `CosyVoice2` were
rejected despite being easier to install: **they appear in zero baseline papers**,
so their numbers would be incomparable.

### B1. DiffVC — primary
* Popov et al., *Diffusion-Based Voice Conversion with Fast Maximum Likelihood
  Sampling Scheme*, **ICLR 2022 (Oral)**, arXiv 2109.13821.
* Repo `github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC`, **MIT** (licence
  lives in the `DiffVC/` subdirectory; the repo-root LICENSE 404s).
* Checkpoints are **Google Drive** links in the README, not HuggingFace — use
  `gdown` to survive the large-file confirmation-token problem. Needed: LibriTTS VC
  model, universal HiFi-GAN vocoder, LibriTTS encoder, speaker encoder.
* **Native 22.05 kHz.** Our pipeline is 16 kHz, so resampling is required in both
  directions. Log it as a processing step — it is a potential confound and must be
  applied identically to protected and control audio.
* Inference is notebook-driven (`inference.ipynb`); port to a script.
* Training wants Montreal Forced Aligner — **not needed for inference, skip it.**
* It is also the model De-AntiFake chose for its adaptive-protection experiment,
  which is the experiment closest to ours. That makes it the most directly
  comparable target we can pick.

### B2. OpenVoice V2 — second
* **The official download bucket is dead** — the MyShell S3 endpoint returns
  `NoSuchBucket` for both V1 and V2. Use the HuggingFace mirrors
  `myshell-ai/OpenVoice` and `myshell-ai/OpenVoiceV2`. Record this in the paper's
  reproducibility note; anyone following the official README will fail.
* De-AntiFake uses **V2**, SafeSpeech uses **V1**. Prefer V2 for De-AntiFake
  comparability, and say which.

### B3. Fix the cloning-success threshold ❌ BLOCKER FOR ANY CLONING NUMBER
`evaluation/cloning_attack.py:61` hardcodes `CLONE_SUCCESS_THRESHOLD = 0.75`,
commented as "standard speaker-verification threshold". **It matches no baseline
paper and contradicts our own EER-calibrated thresholds** (ECAPA 0.29–0.49
depending on channel).

The field convention is **cosine 0.25 on ECAPA-TDNN** — used by SafeSpeech,
VoiceCloak and ClearMask, with De-AntiFake using 0.25 on x-vector. It originates in
Jiang et al., *Mega-TTS 2*, ICLR 2024; cite that if we adopt it.

**Done =** the constant is deleted; thresholds come from
`calibrate_thresholds.py` through the actual channel (our stricter existing
practice), **and** we additionally report at the 0.25 convention so the numbers are
comparable both ways.

### B4. Replicate AntiFake's pre-filter
AntiFake evaluates only on samples that cloned successfully **without** protection:
*"AERR is defined as the rate at which DeepFake samples can no longer bypass the
corresponding SV system, given that all samples have already been filtered to bypass
SV without AntiFake."* Without this filter our rates are not comparable to theirs —
an unprotected clone that was going to fail anyway would count as a success.

### B5. Report VC and TTS separately
VocalBridge separates them (Table I = TTS, Table II = VC) and it is the cleanest
treatment of the six. Since we are running two VC systems and no TTS, the honest
framing is **"evaluated against voice conversion; TTS not tested"** — stated, not
implied by omission.

### B6. Add NISQA
**NISQA is the field standard** (AntiFake, De-AntiFake, VocalBridge, VoiceCloak —
4 of 6). We report UTMOS, which only SafeSpeech uses. **DNSMOS is used by none.**
Keep UTMOS, add NISQA. AntiFake's calibration is worth quoting: *"a MOS of 3 or
higher represents relatively good speech quality... the mean MOS for the TIMIT
corpus is measured at 3.45±0.52."*

### B7. The reviewer question we should answer before it is asked
AntiFake's encoder ablation shows a **near-perfect inversion**: a perturbation
crafted against the AdaIN encoder achieves 99.6% on AdaptVC and **0%** on YourTTS;
GE2E achieves 100% on SV2TTS and 4.8% on AdaptVC. **Protection effectiveness is
governed by whether the surrogate encoder matches the target system's speaker
encoder.**

We craft against a three-encoder ensemble (WavLM-SV, ECAPA, d-vector), which
mitigates this, but a reviewer will ask for the ablation. Note the structural echo:
this is *encoder* mismatch, and our §5 result is *purifier* mismatch — the same
phenomenon one level up. Worth a sentence in the discussion rather than treating
them as unrelated.

---

## C. CODE FIXES

### C1. `evaluation/cloning_attack.py` — threshold
See B3. Also note the file has **two modes**, and only the first is a proxy: mode 1
is an embedding-space attack, **mode 2 (lines ~126–260) is a real YourTTS cloning
path** via Coqui `TTS.api`. It is unused and untested but functional. Either wire it
up as a third (TTS) system or delete it — leaving a working-but-unvalidated cloning
path in the repo invites someone to run it and trust the output.

### C2. `models/cloning_proxy.py` — delete or quarantine
A "surrogate cloning model" that is a WavLM encoder plus a shallow MLP stub. It
feeds nothing and appears nowhere in the results. Its name makes it look like a
cloning test. **Delete it**, or move it to an `unused/` directory with a README
saying it was never used for any published number.

### C3. Audit the verdict helpers
Two automated verdicts were confidently wrong today:
* `exp_vocoder_control.py` credited the purifier with the vocoder's own 1.27×
  head start and printed "TRANSFERS" off an uncorrected ratio.
* `analyse_mismatch.py` branched on the p-value without checking the **sign** and
  announced a large, highly significant *deficit* as "GENERIC ROBUSTNESS".

Both fixed with the reasoning recorded in the code. **Rule: these helpers are for
scanning, never for quoting.** Audit any remaining verdict block for the same two
failure modes — missing baseline, missing sign check.

### C4. Dead-code sweep
`utils/csvio.py` exists because `pandas` silently parses the string `null` as NaN,
which was deleting the zero-noise direction family from every CSV re-read. Confirm
every analysis path uses it.

---

## D. EXPERIMENTS STILL OWED

| # | experiment | cost | why |
|---|---|---|---|
| D1 | Protection replicated on utterance 6 | ~2.2 GPU-h | The negative result is 20 speakers × **1** utterance; survival has two |
| D2 | `range` directions added to E9 | ~30 min | Would extend the spectrum-transfer claim (ρ=0.997) to foreign purifiers, not just the surviving end |
| D3 | Parametric probe with α | ~20 min | Last piece of A1 |
| D4 | v3→v6 transfer already done ✅ | — | ρ = 0.994, 20/20. Was thought lost; it is on disk |

---

## E. ACCEPTED LIMITATIONS — state them, do not try to fix

1. **Crafting uses a mel-domain surrogate purifier.** Foreign purifiers are used for
   *evaluation* only. Crafting through DiffWave would need gradients through a
   waveform model — out of scope.
2. **n = 20 speakers.** Small for a speech venue, but every claim is paired and
   most sit at the p-floor for that n.
3. **d-vector shows no mismatch effect** in any of the four foreign cells. In the
   body, not the limitations.
4. **Griffin-Lim / phase vocoders for our own chain.** The foreign-purifier arm
   needs no vocoder and is correspondingly stronger; say so.
5. **E10 is one day's work on two purifiers of the same architecture family.** The
   geometry (§4) is far more thoroughly controlled than the headline trade result
   (§5). If §5 draws fire, §4 is the fallback paper.

---

## F. EXECUTION ORDER

Ranked by *how much acceptance probability it buys per hour*, respecting
dependencies. Stop-and-reassess gates are marked — do not run past a failed gate,
because everything after it would be meaningless.

### TIER 0 — running, needs nothing from you
| | task | cost |
|---|---|---|
| 0.1 | **A1** alpha rerun (E4 u3 done, u6 running, probe queued) | self-completing |

### TIER 1 — do first. Cheap, and everything downstream is invalid without them.
| | task | cost | why first |
|---|---|---|---|
| 1.1 | **B3** delete the hardcoded 0.75 threshold; use EER-calibrated + report at 0.25 | ~15 min | **Blocks every cloning number.** A wrong threshold does not crash — it silently produces a full table of meaningless results |
| 1.2 | **B1a** DiffVC: `gdown` the four checkpoints, port `inference.ipynb` to a script, wire 16↔22.05 kHz resampling | 1–2 h | The primary new experiment |
| 1.3 | **B1b** ⛔ **GATE: does UNPROTECTED audio actually clone through DiffVC?** Measure speaker similarity of clone-from-clean vs original | ~20 min | If clean audio does not clone, every protection number downstream is meaningless. This is the cloning analogue of the purifier gain gate, and it is the same mistake we already made once with a purifier that did not purify. **If this fails, stop and diagnose — do not proceed to 2.1** |

### TIER 2 — the result that decides the paper's fate
| | task | cost | why |
|---|---|---|---|
| 2.1 | **B4** AntiFake pre-filter, then protection through DiffVC (4 variants × 20 speakers) | ~2 h | This is the number that closes the "no cloning model" objection |
| 2.2 | ⛔ **GATE: reassess.** If protection holds → the paper is close to done. If it collapses → the thesis needs rewriting *again*, and better now than in review | — | Budget a pause here |
| 2.3 | **B2** OpenVoice V2 (HF mirror — **official bucket is dead**) as the second VC | 1–2 h | One cloning system cannot carry the claim, same logic as one foreign purifier |

### TIER 3 — makes the numbers comparable to the baselines
| | task | cost | why |
|---|---|---|---|
| 3.1 | **B6** add NISQA alongside UTMOS | ~30 min | 4 of 6 baselines use NISQA; **none** uses DNSMOS. Without it our quality numbers float free |
| 3.2 | **D3** parametric probe with α | ~20 min | Completes A1 — one survival metric across the whole paper |
| 3.3 | **B5** separate VC/TTS tables; state plainly that TTS was not tested | writing | Prevents a strong VC result implying a TTS claim we did not make |

### TIER 4 — strengthens; none of it blocks
| | task | cost | why |
|---|---|---|---|
| 4.1 | **D2** add `range` directions to E9 | ~30 min | Extends the spectrum-transfer claim (ρ=0.997) to foreign purifiers, not just the surviving end. Cheap and it could fail, which is what makes it worth running |
| 4.2 | **D1** protection replicated on utterance 6 | ~2.2 h | Negative result is 20 spk × 1 utt; survival has two |
| 4.3 | **B7** encoder ablation (craft against each encoder alone) | ~1.5 h | Pre-empts the AntiFake question. **Most valuable *after* cloning works**, because the real question is whether encoder mismatch predicts *cloning* failure — and that ties it to our purifier-mismatch result |

### TIER 5 — hygiene. Do before shipping, not before experimenting.
| | task | cost |
|---|---|---|
| 5.1 | **C1** wire up or delete the unused YourTTS path in `cloning_attack.py` | ~15 min |
| 5.2 | **C2** delete/quarantine `models/cloning_proxy.py` | ~5 min |
| 5.3 | **C3** audit remaining verdict helpers for missing-baseline and missing-sign-check | ~30 min |
| 5.4 | **C4** confirm every analysis path uses `utils/csvio.py` | ~10 min |

### If time runs short
Tiers 1–2 alone make a submittable paper with a real cloning result on one system.
Tier 3 makes it *comparable*. Tier 4 makes it *hard to attack*. Tier 5 is polish.

**The one thing never to skip is gate 1.3.** Every serious error in this project —
the purifier that did not purify, the identity-destroying channel, the baseline
with cosine 1.000 — was a case of measuring something real against something
broken, and each was caught by a gate exactly like it.


---

## G. TIER 1 EXECUTION LOG (2026-09-16)

### 1.1 threshold ✅
`CLONE_SUCCESS_THRESHOLD = 0.75` deleted. Both attack modes now **require** an
explicit threshold (`ValueError` if omitted), and every result additionally carries
`blocked_at_field_convention` against 0.25 on ECAPA.

### 1.2 DiffVC integrated ✅
126.3 M params, 734 MiB VRAM, ~12 s per 4 s clone. Code MIT, checkpoints from the
README's Google Drive links via `gdown`. **Four compatibility fixes**, each patched
in place with a comment:

| problem | fix |
|---|---|
| `model.postnet` not among the files we first pulled | fetched the extra submodule |
| `from models import Generator` resolved to THIS project's `models/` package | load HiFi-GAN by explicit file path — `sys.path` order must not decide |
| `np.bool` / `np.int` removed in NumPy >= 1.24 | builtins (2 files) |
| `librosa.feature.melspectrogram` positional args | keywords (librosa >= 0.10) |

### 1.3 GATE ⛔ FAILED, then PASSED after two fixes ✅

**First run failed, and in the quiet way** — the clone resembled the SOURCE speaker
more than the victim on all three encoders. Had the gate not existed, the protection
experiment would have reported near-total protection against DiffVC, entirely as an
artefact of a broken attacker.

Diagnosis by decomposition, exactly as E6 separated vocoder from purifier:

| stage | WavLM | ECAPA | d-vec |
|---|---|---|---|
| reference, raw | 0.952 | 0.700 | 0.870 |
| reference through DiffVC's own mel + HiFi-GAN | 0.938 | 0.678 | 0.865 |
| clone (broken) | 0.595 | 0.123 | 0.566 |

The front end was near-lossless, so the fault was in conversion. Two causes, **both
mine**:

1. **Speaker-embedding sample rate.** `encoder.audio.preprocess_wav()` only
   resamples when `source_sr` is passed; given a bare array it ASSUMES 16 kHz. I fed
   it the 22.05 kHz resampled copy, so the encoder read every voice as if
   pitch-shifted by 1.38x. The notebook avoids this by passing a *file path*, which
   librosa resamples automatically — so transcribing notebook code to an array API
   silently broke it. **Lesson: when porting from a notebook, the I/O boundaries are
   where the assumptions hide.**
2. **My "bug fix" to spectral subtraction.** Upstream computes `mel_synth_np` and
   `mel_source_np` from the same tensor, making the subtraction a near no-op. I
   "corrected" it to pass the genuinely distinct source mel; that subtracts real
   energy and pushed all three encoders into leaking source identity. The checkpoint
   was trained and tuned with upstream's behaviour, so matching it is what matters
   regardless of whether the line is a typo. Default is now notebook-faithful.

**After the fixes — GATE PASSED:**

| encoder | chance | clone | ceiling | position |
|---|---|---|---|---|
| ECAPA-TDNN | 0.071 | **0.395** | 0.676 | 53.5 % |
| Resemblyzer d-vector | 0.549 | **0.775** | 0.843 | 76.8 % |
| WavLM-SV | 0.608 | **0.892** | 0.951 | 82.9 % |

8/10 clones exceed the 0.25 convention; the source-leak diagnostic is clean.

### Two further errors caught in the 2.1 smoke tests

* **The 0.25 convention is ECAPA-ONLY.** Chance floors are 0.071 (ECAPA) but 0.549
  (d-vector) and 0.608 (WavLM) — a 0.25 threshold sits *below chance* for those two,
  so every clone trivially "succeeded". Now per-encoder EER thresholds, with 0.25
  kept for ECAPA for comparability.
* **The pre-filter must be per encoder.** AntiFake filters against "the
  corresponding SV system"; a single ECAPA mask would score speakers on encoders
  where the attack never worked.

**Pattern worth naming:** every one of these is the same error — measuring something
real against something broken, or against a scale that does not apply. The gates are
catching what code review would not.
