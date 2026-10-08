"""core/matched_random.py -- controls for "is it really a *subspace*?"  (PLAN_V3 W1 #3)

THE DEFECT THIS FIXES
---------------------
Every alpha in RESULTS section 32b is read against ONE baseline: an isotropic
white-noise mel direction (`exp_nullspace_survival.py`, `exp_encoder_subspace.py`
both draw `randn` and scale it).  Council item #3 (2026-10-04, reviewer R1):

    the purifier is a denoiser TRAINED TO REMOVE WHITE NOISE, so white noise is
    the single worst possible reference point.  The vocoder round trip alone
    already buys 1.36x (section 11b).  "encoder identity directions live outside
    the purifier's invariant subspace" may be nothing more than "smooth and
    band-limited things survive, and encoder directions are neither".

C1 is the claim that alpha ORDERS the encoders.  If the ordering survives only
against white noise, C1 is a statement about spectral smoothness wearing a
subspace costume.  So each encoder direction needs a random control that is
matched to it on the property we are NOT claiming, and random on the property we
ARE claiming.

THE TWO CONTROLS
----------------
`band_matched(d)`
    White noise rescaled band by band so that its per-mel-band power profile is
    identical to d's, then renormed so ||control|| == ||d|| exactly.  Matches:
    total norm, per-band power, hence the spectral tilt and any band-limiting.
    Randomises: the direction within each band, and all structure across bands
    and across time.  A control that survives as well as d means d's survival is
    explained by its spectral envelope alone -- #3 null.

`pca_span(basis, d)`
    Every time frame drawn from the span of the top-k principal components of
    CLEAN log-mel frames, then renormed so ||control|| == ||d||.  Matches: the
    low-dimensional "shapes that clean speech is made of", so this control is as
    smooth and as speech-like as the corpus allows.  Randomises: everything about
    which encoder, if any, reads it.  A control that survives as well as d means
    the invariant subspace is just the clean-speech principal subspace -- a real
    and reportable finding, but NOT a per-encoder one.

The basis is built from a corpus subset DISJOINT from the evaluation speakers
(default dev-clean, while every reported number is test-clean) and cached to
disk, so it is fixed across speakers, encoders and runs, and cannot launder
test-speaker information into a control.

READING (pre-registered here, before the run -- PLAN_V3 section 4 wording)
--------------------------------------------------------------------------
Primary endpoint: mean paired per-speaker difference alpha(g_e) - alpha(control_e),
unit = speaker, n = 29, cell = purifier v7 DDIM t=50, estimator = mean paired
difference with a BCa bootstrap 95 % CI (2000 resamples) and a Wilcoxon
signed-rank p, Holm-corrected across the encoders tested.

  * every encoder direction still BELOW both matched controls, and the
    WavLM > ECAPA > x-vector > d-vector ordering preserved
        -> C1 holds as written.  The controls become a row of Fig. 1 and the
           "trained to remove white noise" objection is answered in one panel.
  * controls land BETWEEN white noise and the encoder directions, ordering
    preserved
        -> C1 holds, with a quantified spectral component: report how much of
           the encoder-vs-random gap the spectral envelope explains.
  * encoder directions NOT below the matched controls, or the ordering gone
        -> #3 NULL.  FRAME_v6 section 6 pre-holds this branch: C1 becomes a
           spectral statement, C3 leads the paper, C2 becomes the mechanism.
           Report it as the honest negative; do not re-run looking for a cell
           where it clears.

No other cell may be read as the gate.

WHAT THE CONTROLS DO NOT MATCH (RESULTS 58, risk 1 -- read before quoting them)
-----------------------------------------------------------------------------
Both `band_matched` and `band_matched_wav` match a TIME-AVERAGED spectral
profile and randomise everything else. In time they are therefore as flat as
white noise whatever the target looks like: measured on the 2026-10-08 cohort,
an encoder direction keeps 77-99 % of its energy in its loudest 5 % of
elements (kurtosis 45-3255), its band-matched control 49-69 % (kurtosis 8-27),
isotropic noise 28 % (kurtosis 3); in the waveform domain the control is
Gaussian to three digits (kurtosis 2.96-3.01). `tests/test_matched_random.py`
pins this.

So `band_matched ~ random` in alpha -- which is what every mel cell shows,
within 0.012 -- means "matching the spectrum did not help the control". It does
NOT mean "the control is as structured as the target", and the control cannot
tell "encoder-specific" from "not noise-like". The comparison supports C1 only
in the direction it is read: the encoder directions sit BELOW isotropic noise
too (22-29 of 29 speakers, every encoder, trained or untrained), so the
control's weakness runs against the claim rather than for it. Where a target
sits ABOVE its control (RESULTS 55 on DiffWave, since withdrawn for a different
reason) this control cannot support a verdict at all.
"""
import os

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "experiments", "results", "clean_mel_pca.pt")

# How many principal components the pca_span control is confined to.  Reported
# in every row beside the alpha it controls, together with the fraction of clean
# log-mel variance the basis captures, so the control's strength is on the record.
DEFAULT_K = 20


def _per_band_rms(d):
    """[*, n_mels, T] -> [n_mels] RMS within each mel band."""
    return d.reshape(-1, d.shape[-2], d.shape[-1]).pow(2).mean(dim=(0, 2)).sqrt()


def band_matched(d, generator=None):
    """White noise with d's per-mel-band power profile and d's exact norm.

    Matched: norm, per-band power (so spectral tilt and band-limiting too).
    Random: within-band direction, cross-band and temporal structure.
    """
    z = torch.randn(d.shape, generator=generator, device=d.device, dtype=d.dtype)
    tgt = _per_band_rms(d)                                  # [n_mels]
    cur = _per_band_rms(z).clamp_min(1e-12)
    shape = [1] * d.dim()
    shape[-2] = d.shape[-2]
    z = z * (tgt / cur).reshape(shape)
    return z * (d.norm() / z.norm().clamp_min(1e-12))


def pca_span(basis, d, generator=None):
    """A direction whose every time frame lies in span(basis), with d's exact norm.

    `basis` is [n_mels, k] with orthonormal columns (see `clean_mel_basis`).
    """
    k = basis.shape[1]
    T = d.shape[-1]
    basis = basis.to(device=d.device, dtype=d.dtype)
    c = torch.randn((k, T), generator=generator, device=d.device, dtype=d.dtype)
    v = (basis @ c).reshape(d.shape)
    return v * (d.norm() / v.norm().clamp_min(1e-12))


def clean_mel_basis(k=DEFAULT_K, subset="dev-clean", n_files=60, max_sec=4.0,
                    cache=CACHE, device=None, verbose=True):
    """Top-k principal directions of CLEAN log-mel frames, cached to disk.

    Built from `subset`, which must be disjoint from the evaluation speakers.
    Eigendecomposition runs on CPU in float64 so the basis is bit-stable across
    machines and the golden-file test is meaningful.

    Returns (basis [n_mels, k] float32, info dict).
    """
    want = dict(k=int(k), subset=str(subset), n_files=int(n_files),
                max_sec=float(max_sec))
    if cache and os.path.isfile(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if {kk: blob["info"][kk] for kk in want} == want:
            basis = blob["basis"]
            if verbose:
                print(f"[matched-random] PCA basis from cache: k={want['k']} "
                      f"{want['subset']} n_files={want['n_files']} "
                      f"var={blob['info']['var_explained']:.3f}")
            return (basis.to(device) if device is not None else basis), blob["info"]

    # Imported here so that `band_matched` stays usable without the corpus.
    from evaluation.evaluate import wav_to_logmel
    from experiments.librispeech import list_librispeech, load_flac

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             subset, min_sec=max_sec)
    if not items:
        raise SystemExit(f"[matched-random] no utterances >= {max_sec}s in {subset}")
    # One utterance per speaker, speakers in id order: a spread over the subset
    # rather than many chapters of whoever sorts first.
    seen, chosen = set(), []
    for it in items:
        if it["speaker"] in seen:
            continue
        seen.add(it["speaker"])
        chosen.append(it)
        if len(chosen) >= n_files:
            break
    if verbose:
        print(f"[matched-random] building clean-mel PCA basis: {len(chosen)} "
              f"utterances from {subset} (disjoint from the eval speakers)")

    cpu = torch.device("cpu")
    frames = []
    for it in chosen:
        mel = wav_to_logmel(load_flac(it["path"], max_sec).to(cpu), cpu)
        frames.append(mel.reshape(mel.shape[-2], mel.shape[-1]).double().T)
    X = torch.cat(frames, dim=0)                       # [n_frames, n_mels]
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)              # ascending
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    if not 0 < k <= evecs.shape[1]:
        raise SystemExit(f"[matched-random] k={k} outside 1..{evecs.shape[1]}")
    var_explained = float(evals[:k].sum() / evals.clamp_min(0).sum())
    basis = evecs[:, :k].contiguous().float()
    info = dict(want, var_explained=round(var_explained, 4),
                n_frames=int(X.shape[0]), n_speakers=len(chosen),
                n_mels=int(X.shape[1]))
    if verbose:
        print(f"[matched-random] basis ready: k={k} of {X.shape[1]} mels, "
              f"{var_explained:.1%} of clean log-mel variance, "
              f"{X.shape[0]} frames from {len(chosen)} speakers")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(dict(basis=basis, info=info), cache)
        if verbose:
            print(f"[matched-random] cached -> {cache}")
    return (basis.to(device) if device is not None else basis), info


def controls_for(d, basis, seed, verbose=False):
    """The control set for one target direction d.  Keys name the `kind` column.

    Seeds are derived from `seed` with distinct offsets, so the two controls are
    independent draws and both are reproducible from the speaker id alone.
    """
    out = {}
    g = torch.Generator(device=d.device).manual_seed(int(seed) + 101)
    out["band_matched"] = band_matched(d, generator=g).detach()
    if basis is not None:
        g2 = torch.Generator(device=d.device).manual_seed(int(seed) + 202)
        out["pca_span"] = pca_span(basis, d, generator=g2).detach()
    if verbose:
        for k_, v in out.items():
            print(f"    [ctrl] {k_}: |d|={v.norm():.4f} (target {d.norm():.4f})")
    return out


# ---------------------------------------------------------------------------
# Waveform-domain controls, for council #13c.
#
# alpha is measured on the MEL purifier (v7) while the phenomenon it explains
# -- C2's constraint effect and its d-vector reversal -- lives on DiffWave,
# which is a WAVEFORM model. Explanation and explanandum on different
# purifiers is the one structural hole a reviewer sees in thirty seconds
# (PLAN_V3 W7). Measuring alpha through DiffWave needs waveform directions,
# and therefore a waveform analogue of band_matched.
#
# The construction is the same idea one domain over: match the per-FREQUENCY
# power profile of the target direction's STFT magnitude, randomise phase and
# all cross-bin and temporal structure, then renorm to the target's exact
# L2 norm.
# ---------------------------------------------------------------------------

_WIN_CACHE = {}


def _win(n, device, dtype):
    k = (n, str(device), str(dtype))
    if k not in _WIN_CACHE:
        _WIN_CACHE[k] = torch.hann_window(n, device=device, dtype=dtype)
    return _WIN_CACHE[k]


def band_matched_wav(d, n_fft=512, hop=128, generator=None, n_iter=3):
    """White noise given d's per-frequency power profile and d's exact norm.

    d: [1, L] or [L] waveform-domain direction.

    Matched : total L2 norm, and the average magnitude of every STFT frequency
              bin -- hence the spectral tilt and any band-limiting, which is
              what the 6 kHz bandlimit objection is about.
    Random  : phase, within-bin direction, and all structure across bins and
              across time.

    A control that survives as well as d means d's survival through the
    waveform purifier is explained by its spectrum alone.

    `n_iter` re-imposes the target profile after each inverse transform.  A
    modified spectrogram is generally INCONSISTENT -- no waveform has exactly
    that STFT -- so one pass leaves the profile off, and worst at sharp
    spectral edges, which is exactly where a bandlimited direction lives.

    Measured on a deliberately harsh test (white noise with a 2 kHz cliff,
    100:1), energy-weighted relative error of the per-bin profile:

        n_iter   1       2       3       5
        err      3.1 %   2.1 %   1.8 %   1.4 %

    It converges slowly, so n_iter is not a way to make the match exact; the
    default of 3 is where the returns stop being worth the transforms.  Real
    encoder gradients are far smoother than that test signal.  What does NOT
    depend on n_iter is the property the control exists for: |cos(d, control)|
    stays at ~0.006 throughout, i.e. the control is orthogonal to the target
    whatever the profile error, and the L2 norms match to 7 significant
    figures.
    """
    flat = d.reshape(-1)
    L = flat.shape[-1]
    dev, dt = flat.device, flat.dtype
    w = _win(n_fft, dev, dt)

    D = torch.stft(flat, n_fft=n_fft, hop_length=hop, window=w,
                   return_complex=True, center=True)
    tgt = D.abs().mean(dim=-1).clamp_min(1e-12)              # [F]

    out = torch.randn(L, generator=generator, device=dev, dtype=dt)
    for _ in range(max(1, int(n_iter))):
        Z = torch.stft(out, n_fft=n_fft, hop_length=hop, window=w,
                       return_complex=True, center=True)
        cur = Z.abs().mean(dim=-1).clamp_min(1e-12)          # [F]
        Z = Z * (tgt / cur).unsqueeze(-1)
        out = torch.istft(Z, n_fft=n_fft, hop_length=hop, window=w,
                          length=L, center=True)
    out = out * (flat.norm() / out.norm().clamp_min(1e-12))
    return out.reshape(d.shape)


def controls_for_wav(d, seed, n_fft=512, hop=128, verbose=False):
    """The waveform control set for one direction. Mirrors controls_for().

    `pca_span` has no clean waveform analogue -- the mel PCA basis is over
    log-mel FRAMES -- so it is deliberately absent here rather than faked, and
    #13c reports the two controls it can construct honestly: band_matched and
    isotropic random.
    """
    g = torch.Generator(device=d.device).manual_seed(int(seed) + 101)
    out = {"band_matched": band_matched_wav(d, n_fft, hop, generator=g).detach()}
    if verbose:
        print(f"    [ctrl-wav] band_matched: |d|={out['band_matched'].norm():.4f} "
              f"(target {d.norm():.4f})")
    return out
