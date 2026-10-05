"""
experiments/multi_encoders.py
=============================
Differentiable wrappers for the three speaker encoders, so a single waveform
perturbation can be optimised (PGD) against all of them jointly.

  - WavLM  : the project's WavLMSpeakerEncoder (already differentiable)
  - ECAPA  : speechbrain EncoderClassifier.encode_batch (differentiable)
  - d-vector : Resemblyzer VoiceEncoder fed a TORCH mel front-end that matches
               librosa's mel (slaney scale + slaney norm, power 2.0) so the
               gradient direction matches the REAL eval-time embedding.

Each wrapper exposes .embed(wav[1,T]) -> L2-normalised embedding with grad,
and .name.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
try:
    import torchaudio.transforms as T
except Exception:   # Smart App Control blocks libtorchaudio.pyd (2026-09-15)
    from utils import tacompat as T


def _l2(x):
    return F.normalize(x.float().reshape(1, -1), dim=-1)


class WavLMEnc:
    name = "WavLM (target)"
    threshold = 0.50
    def __init__(self, wavlm):
        self.m = wavlm
    def embed(self, wav):
        w = wav if wav.dim() == 2 else wav.unsqueeze(0)
        return _l2(self.m(w))


class ECAPAEnc:
    name = "ECAPA-TDNN"
    threshold = 0.25
    def __init__(self, clf):
        self.clf = clf
    def embed(self, wav):
        w = wav.reshape(1, -1).float()
        return _l2(self.clf.encode_batch(w))          # grad flows


class DvecEnc:
    """Resemblyzer d-vector with a torch mel front-end matching librosa."""
    name = "Resemblyzer d-vector"
    threshold = 0.75
    def __init__(self, dvec, device):
        from resemblyzer.hparams import (sampling_rate, mel_window_length,
                                         mel_window_step, mel_n_channels,
                                         partials_n_frames)
        self.dvec = dvec
        self._pn = partials_n_frames        # 160 frames = 1.6 s
        self._slice_cache = {}
        n_fft = int(sampling_rate * mel_window_length / 1000)   # 400
        hop = int(sampling_rate * mel_window_step / 1000)        # 160
        # librosa.feature.melspectrogram defaults: power=2.0, slaney mel, slaney norm
        self.mel = T.MelSpectrogram(
            sample_rate=sampling_rate, n_fft=n_fft, win_length=n_fft,
            hop_length=hop, n_mels=mel_n_channels, power=2.0,
            norm="slaney", mel_scale="slaney", f_min=0.0, f_max=sampling_rate / 2,
        ).to(device)
    # ------------------------------------------------------------------
    # FIX P0 (2026-09-19). The crafting path must be the SCORING path.
    #
    # The old embed() ran one forward pass over the whole clip and returned a
    # single pooled embedding. The real evaluation path, `real_embed_dvec`, calls
    # Resemblyzer's `embed_utterance`, which SPLITS the clip into 1.6 s partials,
    # embeds each, and returns the L2-normed AVERAGE.
    #
    # Measured gap (experiments/diag_dvec_proxy.py, 6 test-clean speakers):
    #     pooled vs real scoring path      cos 0.882
    #     pooled vs partial-averaged       cos 0.872   <- the cause
    #     partial-averaged vs real         cos 0.977   <- preprocessing is minor
    #
    # So we were optimising a function ~0.13 cosine away from the one that scores
    # us, which is why Resemblyzer sat at 35-50 % clone success in the DEPLOYMENT
    # setting while WavLM-SV and ECAPA reached 0 % -- it was being crafted against
    # in name only.
    #
    # AntiFake (CCS 2023) Sec. C2 identifies this exact failure and its cause:
    #   "directly deviating the final embedding is susceptible to local optima, as
    #    the impact of a partial embedding not close to the target is mitigated by
    #    other partials that may have attained better optimal points."
    # Their fix is a weighted segment-based deviation. `embed_partials` below
    # exposes the per-partial embeddings so craft() can push the WORST one instead
    # of letting good partials mask bad ones.
    # ------------------------------------------------------------------
    def _mel_frames(self, wav):
        """[frames, 40] log-free mel, matching Resemblyzer's librosa front-end."""
        w = wav.reshape(1, -1).float()
        return self.mel(w).squeeze(0).transpose(0, 1)          # [frames, 40]

    def _slices(self, n_samples):
        # embed_utterance's defaults; cached because the clip length is fixed.
        key = int(n_samples)
        if key not in self._slice_cache:
            self._slice_cache[key] = self.dvec.compute_partial_slices(
                key, rate=1.3, min_coverage=0.75)
        return self._slice_cache[key]

    def embed_partials(self, wav):
        """[n_partials, 256], each L2-normed. Differentiable."""
        w = wav.reshape(1, -1).float()
        wav_slices, mel_slices = self._slices(w.shape[-1])
        need = wav_slices[-1].stop - w.shape[-1]
        if need > 0:                       # embed_utterance zero-pads the same way
            w = F.pad(w, (0, need))
        mel = self._mel_frames(w)                               # [frames, 40]
        nf = mel.shape[0]
        parts = []
        for s in mel_slices:
            seg = mel[s]
            if seg.shape[0] < self._pn:    # last partial can run past the mel
                seg = F.pad(seg, (0, 0, 0, self._pn - seg.shape[0]))
            parts.append(seg)
        parts = torch.stack(parts)                              # [n, 160, 40]
        return F.normalize(self.dvec(parts), dim=-1)            # [n, 256]

    def embed(self, wav):
        """The SAME quantity embed_utterance returns: L2-norm of the mean of the
        L2-normed partial embeddings."""
        return _l2(self.embed_partials(wav).mean(dim=0))


def _stub_speechbrain_optional_deps(max_tries=12):
    """SpeechBrain resolves model classes through `pydoc.locate`, which walks the
    module tree and triggers LAZY imports of optional integrations. Three of them
    are absent here (k2_fsa, huggingface.wordemb, nlp) and the import fails, so
    `spkrec-xvect-voxceleb` will not load at all.

    None of those are used by a speaker encoder: k2 is a finite-state-automaton
    library for ASR lattices, wordemb and nlp are text tooling. An x-vector is a
    TDNN over filterbank features. We insert empty modules so the scan completes.

    This is deliberately narrow: it only stubs a name SpeechBrain itself asked for
    and that is genuinely missing, it stops after `max_tries`, and it returns the
    list so a caller can log exactly what was faked. Installing k2 on Windows to
    load a TDNN would be the wrong trade.
    """
    import re, types
    stubbed = []

    def attempt(fn):
        for _ in range(max_tries):
            try:
                return fn(), stubbed
            except Exception as ex:
                m = re.search(r"target=([\w\.]+)", str(ex))
                if not m or m.group(1) in stubbed:
                    raise
                name = m.group(1)
                sys.modules[name] = types.ModuleType(name)
                stubbed.append(name)
        raise RuntimeError(f"still failing after {max_tries} stubs: {stubbed}")
    return attempt


class XVectorEnc:
    """SpeechBrain x-vector (TDNN, 512-d, VoxCeleb).

    Added 2026-09-20 as the first extension of the crafting ensemble. Chosen
    first because (a) it is architecturally distinct from both current encoders
    -- a TDNN over filterbanks, where WavLM-SV is a transformer over raw waveform
    and ECAPA is a channel-attentive CNN -- and (b) De-AntiFake scores with an
    x-vector system, so attacking one is directly relevant to that baseline.

    AntiFake Table 4 is the evidence that ensemble size is the lever: crafting
    against one encoder protects the synthesizer built on it at 99.6-100 % and
    others at 0-20.6 %; crafting against four reaches 97-100 % everywhere. Their
    Sec. 7 also concedes they "did not identify any voice feature that was
    generalizable across all encoders" -- so this buys COVERAGE, not a universal
    feature, and the paper should say it that way.
    """
    name = "x-vector"
    threshold = 0.25          # placeholder; real value comes from calibration

    def __init__(self, clf):
        self.clf = clf

    def embed(self, wav):
        return _l2(self.clf.encode_batch(wav.reshape(1, -1).float()))


def build_xvector(device, savedir):
    from speechbrain.inference.speaker import EncoderClassifier
    attempt = _stub_speechbrain_optional_deps()
    clf, stubbed = attempt(lambda: EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=os.path.join(savedir, "xvect"),
        run_opts={"device": str(device)}))
    if stubbed:
        print(f"[enc] x-vector loaded; stubbed absent optional deps: "
              f"{', '.join(stubbed)}")
    for p in clf.mods.parameters():
        p.requires_grad_(False)
    return XVectorEnc(clf)


def real_embed_xvector(enc, wav):
    with torch.no_grad():
        return enc.embed(wav)


def build_all(device, wavlm, savedir):
    """Return list [WavLMEnc, ECAPAEnc, DvecEnc]."""
    from speechbrain.inference.speaker import EncoderClassifier
    from resemblyzer import VoiceEncoder
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(savedir, "ecapa"), run_opts={"device": str(device)})
    dvec = VoiceEncoder(device=device)
    return [WavLMEnc(wavlm), ECAPAEnc(ecapa), DvecEnc(dvec, device)]


# Real (eval-time) embedding functions — used for honest evaluation, NOT for PGD
def real_embed_wavlm(wavlm, wav):
    w = wav if wav.dim() == 2 else wav.unsqueeze(0)
    with torch.no_grad():
        return _l2(wavlm(w))

def real_embed_ecapa(clf, wav):
    with torch.no_grad():
        return _l2(clf.encode_batch(wav.reshape(1, -1).float()))

def real_embed_dvec(dvec, wav):
    """Real resemblyzer path: preprocess_wav (librosa) + embed_utterance."""
    from resemblyzer import preprocess_wav
    arr = wav.detach().squeeze().cpu().numpy().astype(np.float32)
    pp = preprocess_wav(arr, source_sr=16000)
    return torch.tensor(dvec.embed_utterance(pp)).reshape(1, -1)


class WavLMSVEnc:
    """microsoft/wavlm-base-plus-sv: WavLM-Base-Plus with a TRAINED x-vector
    head (VoxCeleb1). Replaces the project's random-head WavLM as the real
    speaker-verification target. Model card suggests cosine threshold 0.86."""
    name = "WavLM-SV"
    threshold = 0.86

    def __init__(self, device):
        from transformers import WavLMForXVector
        self.m = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv").to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def embed(self, wav):
        w = wav.reshape(1, -1).float()
        # Wav2Vec2FeatureExtractor(do_normalize=True): zero-mean, unit-var
        w = (w - w.mean()) / torch.sqrt(w.var() + 1e-7)
        return _l2(self.m(input_values=w).embeddings)


def real_embed_wavlm_sv(enc, wav):
    with torch.no_grad():
        return enc.embed(wav)
