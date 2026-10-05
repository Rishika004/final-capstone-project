"""
models/diffvc_clone.py
======================
DiffVC as a voice-CLONING attacker, wrapped to a single callable.

Popov, Vovk, Gogoryan, Sadekova, Kudinov, Wei, "Diffusion-Based Voice Conversion
with Fast Maximum Likelihood Sampling Scheme", ICLR 2022 (Oral), arXiv:2109.13821.
Code and weights from github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC,
MIT (the licence lives in the DiffVC/ subdirectory; the repo root has none).

WHY DiffVC
----------
A verified sweep of the baseline papers (paper/related_work.md) found DiffVC used
by De-AntiFake, VoiceCloak and ClearMask -- more than any other system we could
actually run -- and it is the model De-AntiFake chose for its adaptive-protection
experiment, the one closest to ours. Easier options (kNN-VC, FreeVC, SpeechT5)
appear in *zero* baseline papers, so their numbers would be incomparable.

THE THREAT MODEL, WHICH DECIDES WHAT GOES WHERE
-----------------------------------------------
Voice conversion clones by *conditioning on a reference clip of the victim*:

    reference / target = the audio the victim PUBLISHED (protected or not)
    source             = the attacker's own speech, supplying the words
    output             = the attacker's words in the victim's voice

So the protection is applied to the REFERENCE, never to the source. Success is
measured as speaker similarity between the converted output and the victim's
ORIGINAL clean voice -- i.e. did the attacker obtain the victim's identity.

TWO THINGS TO KNOW BEFORE TRUSTING THE OUTPUT
---------------------------------------------
1. **Sample rate.** DiffVC is 22.05 kHz; this project is 16 kHz. We resample in
   and out. That is a processing step applied identically to protected and control
   audio, but it is a confound to declare, not to hide.
2. **DiffVC's speaker encoder IS the Resemblyzer checkpoint.** Not "the same
   family" -- the same bytes: `checkpoints/diffvc/spk_encoder/pretrained.pt` and
   `resemblyzer/pretrained.pt` are both MD5 a2e2a0bf4cc3ac59053d3fe4d67b0c4b,
   17,090,379 bytes (re-verified 2026-09-17). The encoder and vocoder checkpoints
   differ, so it is specifically the speaker pathway that is shared.

   This is NOT documented anywhere upstream. The DiffVC paper attributes its
   speaker network only to Jia et al. 2018 and never names GE2E, Resemblyzer or
   Jemine; what the repo documents is shared CODE lineage (`speaker_encoder/` files
   carry `from https://github.com/CorentinJ/Real-Time-Voice-Cloning`), not shared
   checkpoint identity. Anyone re-deriving this should use the public git blob
   SHA-1 rather than downloading and hashing: both repos store the file in-tree, and
   GitHub reports 47e405a459e4b97a21325dc78ba6f8270498ac06 / 17090379 bytes for
   both paths.

   AntiFake Table 4 is the prior for what that buys. Crafting against one encoder
   protects the synthesizer built on that encoder at 99.6-100 % and the other four
   at 0-20.6 %; their GE2E row -- our case -- is 100 % on the GE2E-based synthesizer
   against 1.2-20.6 % elsewhere. So our d-vector arm is MATCHED to this attacker
   while WavLM-SV and ECAPA are not. Expect that to show up in the results, and
   report it rather than letting it look like noise. `--craft-encoders no-dvec` on
   `exp_purifier_aware.py` is the ablation that separates the two (fix F1).

Usage:
    from models.diffvc_clone import DiffVCCloner
    vc = DiffVCCloner(device="cuda")
    wav16 = vc.clone(source_wav16, reference_wav16)
"""
import os
import sys

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DVC = os.path.join(PROJ, "models", "diffvc")
for _p in (_DVC, os.path.join(_DVC, "hifi-gan"), os.path.join(_DVC, "speaker_encoder")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CKPT = os.path.join(PROJ, "checkpoints", "diffvc")
DIFFVC_SR = 22050
PROJECT_SR = 16000


def _resample(wav, sr_from, sr_to):
    if sr_from == sr_to:
        return wav
    import torchaudio.functional as AF
    return AF.resample(wav, sr_from, sr_to)


class DiffVCCloner:
    def __init__(self, device="cuda", n_timesteps=30, mode="ml"):
        import json
        import librosa
        from librosa.filters import mel as librosa_mel_fn
        import params
        from model import DiffVC
        from env import AttrDict
        # HiFi-GAN's module is also called `models`, which collides with THIS
        # project's models/ package -- `from models import Generator` resolves to
        # ours and fails. Load it by explicit file path so import order cannot
        # decide which one wins.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "diffvc_hifigan_models", os.path.join(_DVC, "hifi-gan", "models.py"))
        _hfg = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_hfg)
        HiFiGAN = _hfg.Generator
        from encoder import inference as spk_encoder
        from pathlib import Path

        self.device = torch.device(device)
        self.n_timesteps, self.mode = n_timesteps, mode
        self._librosa = librosa
        # DiffVC's own mel front end -- NOT this project's. Do not substitute ours:
        # the model was trained on these exact settings.
        self.mel_basis = librosa_mel_fn(sr=DIFFVC_SR, n_fft=1024, n_mels=80,
                                        fmin=0, fmax=8000)

        gen = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                     params.layers, params.kernel, params.dropout, params.window_size,
                     params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                     params.beta_min, params.beta_max)
        sd = torch.load(os.path.join(CKPT, "vc", "vc_libritts.pt"),
                        map_location="cpu", weights_only=False)
        gen.load_state_dict(sd)
        self.generator = gen.to(self.device).eval()

        with open(os.path.join(CKPT, "vocoder", "config.json")) as f:
            h = AttrDict(json.load(f))
        voc = HiFiGAN(h)
        voc.load_state_dict(torch.load(os.path.join(CKPT, "vocoder", "voc_ckpt.pth"),
                                       map_location="cpu", weights_only=False)["generator"])
        voc = voc.to(self.device).eval()
        voc.remove_weight_norm()
        self.vocoder = voc

        spk_encoder.load_model(Path(os.path.join(CKPT, "spk_encoder", "pretrained.pt")),
                               device=device)
        self.spk_encoder = spk_encoder

    # -- DiffVC's mel, transcribed from inference.ipynb ------------------------
    def _mel(self, wav_np):
        wav = wav_np[: (wav_np.shape[0] // 256) * 256]
        wav = np.pad(wav, 384, mode="reflect")
        stft = self._librosa.core.stft(wav, n_fft=1024, hop_length=256,
                                       win_length=1024, window="hann", center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
        return np.log(np.clip(np.matmul(self.mel_basis, stftm), a_min=1e-5, a_max=None))

    @staticmethod
    def _median_smooth(x, w=5):
        y = np.copy(x)
        x = np.pad(x, w, "edge")
        for i in range(y.shape[0]):
            y[i] = min(x[i + w + 1], np.median(x[i:i + 2 * w + 1]))
        return y

    def _spectral_subtraction(self, mel_synth, mel_source, smoothing_window=1):
        """Transcribed from inference.ipynb.

        NOTE an upstream bug: the notebook computes BOTH `mel_synth_np` and
        `mel_source_np` from the same tensor `mel_`, so its subtraction is very
        nearly a no-op. We keep the function faithful but pass the genuinely
        distinct source mel, which is what the code plainly intends. If a result
        ever needs to be matched bit-for-bit against the notebook, this is the
        line that will differ."""
        mel_diff = mel_synth - mel_source
        mel_diff[mel_diff < 0] = 0
        mel_diff = np.array([self._median_smooth(mel_diff[i], smoothing_window)
                             for i in range(mel_diff.shape[0])])
        out = mel_synth - mel_diff
        return out

    @torch.no_grad()
    def clone(self, source_wav, reference_wav, sr=PROJECT_SR, spectral_subtract=False):
        """source_wav: content (attacker's speech). reference_wav: the victim's
        PUBLISHED audio, protected or not. Both [1, T] at `sr`. Returns [1, T] at `sr`."""
        src = _resample(source_wav.squeeze(0).cpu(), sr, DIFFVC_SR).numpy().astype(np.float64)
        ref = _resample(reference_wav.squeeze(0).cpu(), sr, DIFFVC_SR).numpy().astype(np.float64)

        mel_s = torch.from_numpy(self._mel(src)).float().unsqueeze(0).to(self.device)
        mel_t = torch.from_numpy(self._mel(ref)).float().unsqueeze(0).to(self.device)
        len_s = torch.LongTensor([mel_s.shape[-1]]).to(self.device)
        len_t = torch.LongTensor([mel_t.shape[-1]]).to(self.device)

        # Speaker embedding comes from the REFERENCE -- the victim's published audio.
        #
        # CRITICAL: feed the ORIGINAL-RATE audio, not the 22.05 kHz resampled copy.
        # encoder.audio.preprocess_wav() only resamples when source_sr is given;
        # handed a bare array it ASSUMES 16 kHz (params_data.sampling_rate). Passing
        # the 22.05 kHz version made the encoder read the voice as if pitch-shifted
        # by 1.38x, which corrupted the identity embedding and was why conversion
        # produced neither the victim nor the source (gate_cloning.py, 2026-09-16).
        ref16 = reference_wav.squeeze(0).cpu().numpy().astype(np.float32)
        ref_pre = self.spk_encoder.preprocess_wav(ref16, source_sr=sr)
        emb = torch.from_numpy(self.spk_encoder.embed_utterance(ref_pre)
                               ).float().unsqueeze(0).to(self.device)

        _, mel_out = self.generator.forward(mel_s, len_s, mel_t, len_t, emb,
                                            n_timesteps=self.n_timesteps, mode=self.mode)
        if spectral_subtract:
            m = self._spectral_subtraction(mel_out.cpu().squeeze().numpy(),
                                           mel_s.cpu().squeeze().numpy(), 1)
            mel_out = torch.from_numpy(m).float().unsqueeze(0).to(self.device)

        audio = self.vocoder.forward(mel_out).cpu().squeeze().clamp(-1, 1)
        return _resample(audio.unsqueeze(0), DIFFVC_SR, sr)
