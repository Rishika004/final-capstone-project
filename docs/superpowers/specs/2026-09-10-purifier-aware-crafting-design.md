# Purifier-aware perturbation crafting — design (2026-09-10)

Target venue: ICASSP 2027 (full-paper deadline 16 Sep 2026).

## Problem

`experiments/results/end2end.csv` shows the current protection does NOT hold
after the attacker runs DiffPure + Griffin-Lim: WavLM/ECAPA cosine returns
above threshold on 5/5 files. Root causes:

1. `exp_end2end.py` crafts plain ensemble PGD; neither the null-space
   projection nor the purifier appears in its loss.
2. The null-space projection lives in mel space against a single-pass
   surrogate; the delivered waveform is `x + delta_wav`, untouched by it.
3. Griffin-Lim destroys d-vector identity even for unprotected audio, so
   that encoder cannot be judged end to end with Griffin-Lim.
4. The project WavLM encoder has a random 768→256 head (not a real ASV).
5. Only 5 utterances, single source.

## Approach (approved by user, option A)

Craft `delta` so that it is adversarial AFTER purification:

    min_delta  LSE_e[ w_e cos(E_e(V(DP_s(mel(x+delta)))), E_e(x)) ]
             + lam_before * LSE_e[ w_e cos(E_e(x+delta), E_e(x)) ]
    s.t.  |STFT(delta)| <= psychoacoustic masking cap, band-limited

* DP_s: the FULL differentiable DDPM DiffPure loop (t*=50), fresh noise
  seed every step (expectation over transformation, EOT). Gradient
  checkpointing per denoiser call keeps it inside 6 GB.
* V: fixed-original-phase differentiable mel→wav (`core/optimizer.py`);
  evaluation uses the attacker's real Griffin-Lim.
* Encoders in the ensemble: WavLM-SV (`microsoft/wavlm-base-plus-sv`,
  trained x-vector head, replaces the random-head WavLM), ECAPA-TDNN
  (speechbrain), Resemblyzer d-vector (torch mel proxy for gradients,
  real path for evaluation).
* Ablation variants in the same script: `plain` (no purifier in loss),
  `surrogate` (single-pass denoiser in loss), `full` (DDPM EOT).
  The null-space projection is reported as a fourth ablation
  (`nullspace`: legacy mel projection delivered via mel→wav).

## Evaluation (end to end, honest)

Per file, per encoder, over `n_eval_seeds` DiffPure seeds:
`cos_before`, `cos_after_purify` (mean over seeds), `cos_ctrl` (unprotected
through the same DiffPure+GL), `clone_fails_after`, `ctrl_identity_kept`,
`defence_valid` = fails AND control kept. Quality: PESQ-WB, STOI, SNR.

Thresholds: calibrated at EER on LibriSpeech dev-clean same/different
speaker trials (`experiments/calibrate_thresholds.py`); model-card defaults
used until calibration runs.

Data: `data/samples` (5 files) for feasibility, then one 4-s utterance per
LibriSpeech dev-clean speaker (40 speakers) for the paper table.

## Files

* `experiments/exp_purifier_aware.py` — crafting variants + end-to-end eval.
* `experiments/calibrate_thresholds.py` — EER thresholds per encoder.
* `experiments/multi_encoders.py` — add `WavLMSVEnc`.
* Results: `experiments/results/purifier_aware.csv`, wavs under
  `experiments/results/purifier_aware/`.

## Success criteria

`full` variant: `defence_valid` on the majority of files for WavLM-SV and
ECAPA with PESQ >= 3.0. If it fails on the 5-sample feasibility run after
tuning, stop and report; fallback venue is Interspeech 2027.
