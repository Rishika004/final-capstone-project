# Improving the d-vector column — what to change, in order

Written 2026-10-05. Sources: a full read of AntiPure (ICCV 2025, arXiv:2509.13922,
v1+v2 end to end) and a line-level map of the crafting path. Every claim below
cites file:line or a paper equation. New base only (§21-32).

**Headline: the three largest levers are misalignments in our own code, not
anything from AntiPure. All three hit the d-vector hardest. Fix them before
adding any new loss term.**

---

## 0. The numbers that frame the problem

From `thresholds_clean_xsess_tc.csv` (29 spk, cross-session):

| encoder | EER | threshold τ | genuine | impostor | **headroom τ − impostor** |
|---|---|---|---|---|---|
| WavLM-SV | 5.2 % | 0.8926 | 0.9406 | 0.6769 | 0.216 |
| ECAPA | 0.62 % | 0.3358 | 0.6232 | 0.0766 | 0.259 |
| **d-vector** | 3.45 % | 0.7029 | 0.8142 | 0.5639 | **0.139** |

Fraction of available headroom the protection actually consumes:
**ECAPA 96 %, WavLM 68 %, d-vector 57 %.** ECAPA is driven onto its impostor
floor (0.086 vs 0.0766). The d-vector stops **0.078 short of τ with 0.078 still
unspent**. That unspent margin is the target of everything below.

Why the purifier flips it: purification recovers a roughly fixed *absolute*
cosine (+0.071 DiffWave, +0.084 PhonePuRe). For ECAPA that is 42 % of its
headroom; for the d-vector it is **51 % of a headroom half the size**. Small
range × worst survival (§32: α 0.406, nsr > 1) compound.

---

## TIER 0 — Misalignments. ~zero cost. Do first.

### T0.1 `best_delta` is selected by RAW COSINE, not margin
`exp_purifier_aware.py:393` (`score = c_before.max().item()`) and **:412**
(`score_acc += c_after.max().item()/n_eot`).

Thresholds span 0.59 across encoders (0.8926 / 0.3358 / 0.7029), so
`argmax(raw cosine) ≠ argmax(margin)`. A delta with WavLM 0.70 (margin −0.19)
and d-vector 0.69 (margin **−0.013**) scores *better* than one with WavLM 0.75
and d-vector 0.64 (margin −0.063). **The ship-selection step can actively
prefer iterates whose d-vector margin is worse.** Over 200 iterates it chooses
the wrong one.

Fix: `(c_before - thr_t).max().item()` and `(c_after - thr_t).max().item()`.
`thr_t` is already in scope (defined :281). **One token each. No extra compute.**

### T0.2 Placeholder thresholds are ~half the d-vector's headroom too permissive
`multi_encoders.py:54` `DvecEnc.threshold = 0.75` vs calibrated **0.7029**
(clean) / 0.6819 (diffpure t50). Also `:43` ECAPA 0.25 vs 0.3358, `:243` WavLM
0.86 vs 0.8926. `main:737` seeds thresholds from these attrs and only overrides
if `--thresholds` is passed.

A 0.047–0.068 slack on an encoder whose *entire* headroom is 0.139 tells the
optimiser to stop pushing exactly where it matters most. 22 scripts do pass
`--thresholds`, so this is a latent trap rather than the cause of shipped
numbers — but it is the size of the gap, and one forgotten flag reproduces it.

Fix: replace with calibrated values, or raise when `--thresholds` is absent.

### T0.3 `tau_lse = 4.0` is hard-coded and unreachable
`craft()` kwarg at **:270**, never passed from `main`. At the achieved margins
(WavLM −0.13, ECAPA −0.25, d-vector −0.06) the soft-max puts only ~60–65 % of
gradient on the d-vector; ~35 % is spent on encoders already far below
threshold. Expose it and sweep {4, 8, 16} — a sharper max concentrates on the
binding encoder, which is the d-vector by construction.

### T0.4 Do NOT "fix" this with `--weights`
`lse` multiplies `w` **inside** the exponent (**:328**), scaling `(c − τ)`. For
an encoder already *below* threshold, a larger weight makes `(c−τ)·w` more
negative and **de-emphasises** it. The shipped default `1,1,1.5` (**:689**) was
never used; all 16 scripts pass `1,1,1`. Reaching for `--weights 1,1,2` gives
the opposite of the intent once the d-vector crosses below τ. Use `tau_lse`, or
a per-encoder hinge (`sum_e relu(c[e] − thr[e] + m)`), not `w`.

---

## TIER 1 — Front-end alignment. The single biggest d-vector lever.

### T1.1 VAD trimming and volume normalisation were never fixed
§23 fixed the *partial-averaging* half (cosine 0.882 → 0.977) and added
`--seg-partials`. The *preprocessing* half remains:

| | crafting (`DvecEnc`, `multi_encoders.py:109-130`) | evaluation (`real_embed_dvec`, :233-238) | **DiffVC's conditioner** (`diffvc_clone.py:192-195`) |
|---|---|---|---|
| normalise to −30 dBFS | **no** | yes | yes |
| webrtcvad mode-3 silence trim | **no** | yes | yes |
| mel front-end | torchaudio | librosa | librosa |

Why it bites hard: at `rate=1.3` the partial hop is ~12 308 samples ≈ **0.77 s**,
and `--max-sec` (never changed from 4.0) gives only **~4 partials**. VAD trimming
of LibriSpeech removes 0.3–0.8 s — **half to a full partial hop**. So the ~4
windows whose worst case `--seg-partials` optimises are *not* the windows the
evaluator scores. With 4 partials a one-window misalignment is a quarter of the
objective. And `normalize_volume(increase_only=True)` rescales the whole signal —
perturbation included — when the crop is below −30 dBFS.

**It bites twice:** DiffVC's speaker conditioner is the byte-identical GE2E
checkpoint and also goes through `preprocess_wav`. So the cloner and the
evaluator both read a trimmed, renormalised signal that was never crafted for.

Fix, and it is differentiable: apply `normalize_volume` (a scalar gain) and a
**frozen** VAD mask computed once from `wav_o` (a fixed 0/1 mask) inside
`DvecEnc.embed_partials`. The only non-differentiable part of `preprocess_wav`
is the VAD decision itself, which can legitimately be frozen on the clean signal.

Cheap pre-check, no GPU: extend `diag_dvec_proxy.py` (it already isolates
front-end vs geometry at :86-99) with `cos(partials-of-trimmed,
partials-of-untrimmed)` and a count of slice boundaries moving >½ hop.

### T1.2 Longer crops give the smooth-max something to work with
`--max-sec` / `--min-sec` (**:653**, **:651**) have never been overridden —
every crafted clip in the project is 4 s. More partials make both
`--seg-partials` and T1.1 better conditioned. Caveat: changes every number, so
treat as a separate arm, not a silent default change.

---

## TIER 2 — What is genuinely portable from AntiPure

### T2.1 The citation gift: AntiPure rejected OUR objective without testing it
This is the most valuable thing in their paper for us. AntiPure derives **two
opposing sufficient objectives** on the serial purify-then-exploit pipeline:

- **Eq. 5 (survival — this is our method):** `δ = argmin ‖Pure(x₀+δ) − (x₀+δ)‖∞`
- **Eq. 6 (sabotage — theirs):** `δ = argmax ‖Pure(x₀+δ) − (x₀+δ)‖∞`

They state the survival reading explicitly: *"By Eq. 5 combined with Eq. 2, we
can approximate Pure(x) ≈ x, allowing Eq. 4 to degenerate into Eq. 2 even under
purification… leads to a robust perturbation that can still target customization
after purification."* Then they decline it on tractability grounds only —
*"joint optimization of two objectives makes the optimal solution hard to
find"*, *"probabilistic diffusion models may result in generated images that are
almost completely uncontrollable at the fine scale required"* — supported by one
qualitative t-SNE and a citation to someone else's failed attempt.

**They never implement, ablate, or report a single number for Eq. 5.** Their
language is uniformly "harder/impractical", never "impossible". So the gap our
method occupies is one the state of the art asserts is intractable and never
tested — and our `reg` term at **:388** is precisely their Eq. 5, in speech, with
a measurement. Cite this in the intro; it converts "isn't this AntiPure for
audio?" into "AntiPure's own rejected branch, tested."

Note their Fig. 2 argument also cuts both ways: it says the *purified output
distribution* is wide — an argument against steering outputs to a precise
target, **not** against a subspace the purifier does not act on.

### T2.2 Put the perceptual budget IN the graph (`masking_penalty` already exists)
AntiPure's real lesson is that its objective lives *inside* the optimisation, on
a perceptually chosen transform domain. Ours does not: `project_delta_to_masking`
runs **after** `opt.step()` under `no_grad` (**:421-422**), so **the optimiser
never sees the constraint** — it is clipped afterwards.

`core/psychoacoustic.py:267 masking_penalty` is written, differentiable, and
**never called anywhere in the repo**. Adding `alpha * masking_penalty(delta,
cap, masker)` at **:413** costs zero extra model passes.

Why this should help the d-vector specifically: §24a found more steps only "buy"
margin by filling the loudness budget, with every perceptual column degrading.
An in-graph budget term lets the optimiser *allocate* the budget instead of
being clipped into it — which is exactly the regime where the encoder with the
least headroom should win the allocation.

### T2.3 Schedule-weighted invariance (their `e^{ᾱ_t − 1}`)
AntiPure weights its frequency term by `e^{ᾱ_t−1}`: full strength as `t→0`,
decaying at high `t`. Two stated reasons — low timesteps are where fine spectral
detail is decided, **and** the weight cancels the `√(1−ᾱ_t)` growth that would
otherwise let high-`t` samples dominate the gradient.

Our `--null-t` is a **single fixed value** (200 in every script, 100 once;
**:668**). So our invariance term is measured at one noise level while the
attacker purifies at t=1, 3, 50, 200. Sampling `t ~ U(1, t_attack)` with a
schedule weight — their `U(1,t^p)` discipline — is a direct, cheap port:
`ab_t` is already indexed from `diffpure.alpha_bar` at **:354**.

### T2.4 Bandlimit: we zero ~19 % of the d-vector's own input
`project_delta_to_masking:238-241` hard-zeroes every bin ≥ 6 kHz (bin 384 of
513). The d-vector's front-end is 40 mels spanning **0–8000 Hz**
(`multi_encoders.py:68`), so 6–8 kHz is its **top ~7–8 of 40 channels — ~19 % of
its input dimensions receive zero perturbation by construction.** WavLM reads
raw waveform and ECAPA 80 fbanks, so the same cut costs them proportionally
less. `--bandlimit` (**:658**) has **never been varied**.

AntiPure's PFG claims high frequencies are where a purifier has least control.
In speech the *perceptual* half of that argument inverts (2–8 kHz carries
fricatives, F2/F3 and most intelligibility), so we cannot simply push energy up
there. But the *d-vector-specific* question is open and free to test: measure the
energy fraction of `g_dvec` above 6 kHz versus `g_ECAPA`, `g_WavLM` (an
`exp_encoder_subspace.py`-style measurement, no crafting). If the d-vector's
identity gradient is disproportionately high-band, the bandlimit is spending
budget away from where the d-vector looks, and raising it trades quality for
exactly the column we need.

### T2.5 "Sharp survives, smooth dies"
Their empirical rule: PhotoGuard fails because of *"blurred boundaries"*, SimAC
loses to Anti-DB, while blocky/sharp perturbations survive; and Table 8's
crop-scale resampling **amplified** their artefacts. Testable on files we
already have: correlate the transient/spectral-flatness structure of δ with its
measured α. If it holds in speech, it is a new, cheap term — and it would
explain the §32 ordering from a different direction.

---

## TIER 3 — Structural, with honest caveats

### T3.1 Measure clone QUALITY. 22 MB, and currently unmeasured.
`exp_cloning_protection.py:280-292` embeds the clone on :281 and lets it go out
of scope. **No clone waveform is ever saved; no UTMOS, NISQA, WER, PESQ is ever
computed on a clone.** CSV columns are speaker-similarity only.

Adding one `sf.write` after **:280** gives 29 spk × 2 arms × 3 purifiers ≈ 174
files ≈ **22 MB**, and `eval_quality_nr.py` (UTMOS at :62-70, Whisper WER at
:116) runs on them unchanged apart from the input directory.

Why it matters for the d-vector: the d-vector column says clones still *verify*.
It does not say they are *usable*. "Protection works by destroying the clone's
intelligibility rather than its identity" is a different claim, and it is
currently unsupported in either direction. **Highest value per GPU-second left
in the project.**

### T3.2 The AntiPure mechanism probably does NOT port — know this before building
AntiPure's payoff is *"leveraging the characteristics of the fine-tuning process
itself to amplify these artifacts"*: DreamBooth/LoRA memorise the artefact over
500 gradient steps, so residual high-frequency speckle grows into "patches of
abnormal textures". **DiffVC has no amplification stage** — it conditions on one
256-d vector (`diffvc_clone.py:194-195`) and the encoder *pools*, discarding
exactly the fine high-band texture AntiPure plants. Also: `DiffVCCloner.clone`
is `@torch.no_grad()` (**:172**) and mixes numpy/librosa, so it is not
differentiable at all — a clone-aware loss term is not available without
rewriting it.

Also note AntiPure's own weakness: Table 3 shows it is **worse than
Anti-DreamBooth under light purification** (FID 54.45 vs 124.62 at 10
iterations), only winning past 30–40. It is a defence that *requires* the
attacker to purify hard. Ours wins unpurified too (6/0/29 %).

### T3.3 Erroneous Timestep Guidance is not implementable against our purifier —
### and implementing it against DiffWave would destroy our transfer claim
Confirmed: `purifier.py:148` `def forward(self, x)` and `:322` `def forward(self,
x)` — **no timestep argument**. v7 is an unconditional x0 regressor; timesteps
enter only through schedule scalars the *defender* chooses. ETG is therefore
impossible against it.

DiffWave is the only time-conditioned, epsilon-predicting purifier
(`diffwave_purifier.py:109-112`), and it is inference-only for exactly three
mechanical reasons: `@torch.no_grad()` at `:97` and `:118`, plus an in-place
`h += part_t` at `models/diffwave/WaveNet.py:76` that would break autograd
(change to `h = h + part_t`).

**But do not do this.** DiffWave is our *unseen foreign purifier*; the transfer
result (§16, §E9) depends on never having crafted through it. Making it a
crafting target converts our strongest surviving claim into a matched-purifier
result. If ETG is ever wanted, it needs a *fourth*, time-conditioned purifier
held separate from the transfer evaluation.

---

## Recommended order

| # | Change | Cost | Expected d-vector effect |
|---|---|---|---|
| 1 | T0.1 margin-based `best_delta` | 2 tokens | direct; removes wrong-iterate selection |
| 2 | T0.2 calibrated thresholds / raise if absent | 10 min | removes 0.047–0.068 of false slack |
| 3 | T0.3 expose + sweep `tau_lse` {4,8,16} | 30 min + ~2 GPU-h on `--variant surrogate` (50× cheaper, never run) | shifts gradient onto the binding encoder |
| 4 | T1.1 VAD + volume-norm in the crafting front-end | ~4 h code, ~3 GPU-h | largest single lever; fixes the evaluator AND DiffVC |
| 5 | T2.2 `masking_penalty` in-graph | ~1 h, zero extra passes | better budget allocation |
| 6 | T3.1 save clones + score quality | ~1 h, ~1 GPU-h | new claim, not a d-vector fix |
| 7 | T2.4 bandlimit gradient-energy probe | ~2 h, ~1 GPU-h | decides whether to raise bandlimit |
| 8 | T2.3 schedule-sampled `null_t` | ~3 h, ~4 GPU-h | invariance matched to the attacker's range |
| 9 | T2.5 sharpness-vs-α correlation | ~2 h, 0 GPU | possible new term + mechanism support |

Items 1–3 are nearly free and should run as one arm on `--variant surrogate`
before any expensive crafting. **Do not start with a new loss term.**

## Also unexercised, found in the map

`--null-mode directed` (**:664**) is fully implemented at :370-386, its docstring
argues it is the *correct* objective, and it has **never been run once**.
`--variant surrogate` (never run, ~50× cheaper) is the natural sweep rig.
`--craft-stride` (**:624**, documented ~5× speedup, `validate_stride.py` exists)
and `--n-eot` (**:671**, default 1) combine: strided DDIM at `n_eot=3` may cost
less than today's `stride=1, n_eot=1` while closing the EOT-parity objection.
