"""
models/diffwave_purifier.py
===========================
A DiffPure-style purifier built on an UNCONDITIONAL DiffWave, for use as a
FOREIGN purifier: someone else's architecture (WaveNet on raw waveform, not our
2-D mel UNet), someone else's weights, someone else's training data.

Why this file exists
--------------------
E2 showed the pass-through subspace transfers from our purifier v3 to v4. But v3
and v4 are two runs of one recipe. To say anything about *speech diffusion
purifiers* rather than *our purifier*, the evaluation model has to be foreign in
every respect. Two public checkpoints share this exact architecture:

  SC09      philsyn/DiffWave-unconditional, MIT, trained on Speech Commands
            digits (1 s clips). Narrow prior -- a null result against this one
            alone could be a domain artefact rather than a real finding.
  LibriSpeech  De-AntiFake (Fan et al., ICML 2025) `purification.pkl`, weights
            MIT on HuggingFace. This is the purifier from the paper that
            motivates ours, trained on our corpus. The one that matters.

Provenance and licensing
------------------------
The network definition is philsyn's `WaveNet.py` (MIT, vendored under
`models/diffwave/`). The purification loop below is written here from the DDPM
equations and the published schedule (T=200, beta linear 1e-4 -> 0.02) rather
than copied from AudioPure's repo, which ships no LICENSE file. Cite:
  Kong et al., DiffWave, ICLR 2021, arXiv:2009.09761
  Wu et al., Defending against Adversarial Audio via Diffusion Model (AudioPure),
    ICLR 2023, arXiv:2303.01507   -- the forward-to-t-then-reverse recipe
  Nie et al., DiffPure, ICML 2022, arXiv:2205.07460
  Fan et al., De-AntiFake, ICML 2025, arXiv:2507.02606  -- the LibriSpeech weights

Note on t
---------
This model's t is NOT our t. Its schedule is its own (T=200, beta_T=0.02) and
AudioPure purifies at n* = 3 by default, sweeping {0,1,2,3,5,7,10} -- far smaller
than the t=50..300 we use, because the noise is added in the waveform domain
where it is far more destructive. Choose t by matching the DAMAGE the purifier
does to unprotected audio, never by matching the integer.
"""
import os
import sys

import torch

_DW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffwave")
if _DW not in sys.path:
    sys.path.insert(0, _DW)          # WaveNet.py does a bare `from util import ...`

WAVENET_CONFIG = dict(in_channels=1, res_channels=256, skip_channels=256,
                      out_channels=1, num_res_layers=36, dilation_cycle=12,
                      diffusion_step_embed_dim_in=128,
                      diffusion_step_embed_dim_mid=512,
                      diffusion_step_embed_dim_out=512)
T_STEPS, BETA_0, BETA_T = 200, 1e-4, 0.02
NATIVE_SR = 16000


def _schedule(device):
    beta = torch.linspace(BETA_0, BETA_T, T_STEPS, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    # beta_tilde_t = beta_t * (1 - abar_{t-1}) / (1 - abar_t); sigma = sqrt(beta_tilde)
    abar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    sigma = (beta * (1 - abar_prev) / (1 - alpha_bar)).clamp(min=0).sqrt()
    return beta, alpha, alpha_bar, sigma


class DiffWavePurifier(torch.nn.Module):
    """purify(wav, seed) -> wav.  Forward-noise to `t`, then ancestral reverse to 0.

    This is partial diffusion, NOT generation from pure noise: at t=3 of 200 the
    audio is barely perturbed and the reverse loop runs three steps.
    """

    def __init__(self, ckpt_path, t=3, device="cuda", chunk=None):
        super().__init__()
        from WaveNet import WaveNet_Speech_Commands           # noqa: E402 (vendored)
        self.net = WaveNet_Speech_Commands(**WAVENET_CONFIG).to(device).eval()
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"{os.path.basename(ckpt_path)}: {len(missing)} missing "
                               f"keys, e.g. {missing[:3]}")
        self.unexpected = unexpected      # optimizer/EMA state is fine to ignore
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.t = int(t)
        self.device = device
        # The nets are fully convolutional, so length is unconstrained; `chunk`
        # exists only to bound memory on long audio.
        self.chunk = chunk
        self.beta, self.alpha, self.alpha_bar, self.sigma = _schedule(torch.device(device))

    @torch.no_grad()
    def _run(self, wav, seed):
        """wav: [1, L] in [-1, 1] -> [1, L]."""
        x = wav.unsqueeze(0) if wav.dim() == 2 else wav      # -> [B=1, 1, L]
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        t = self.t
        if t <= 0:
            return x.squeeze(0)
        ab = self.alpha_bar[t - 1]
        eps = torch.randn(x.shape, generator=g, device=self.device)
        x = ab.sqrt() * x + (1 - ab).sqrt() * eps            # forward to t only
        for i in range(t - 1, -1, -1):                       # ancestral reverse
            steps = (i * torch.ones((x.shape[0], 1), device=self.device))
            e = self.net((x, steps))
            x = (x - (1 - self.alpha[i]) / (1 - self.alpha_bar[i]).sqrt() * e) \
                / self.alpha[i].sqrt()
            if i > 0:
                x = x + self.sigma[i] * torch.randn(x.shape, generator=g,
                                                    device=self.device)
        return x.squeeze(0)

    @torch.no_grad()
    def forward(self, wav, seed=0):
        if self.chunk is None or wav.shape[-1] <= self.chunk:
            return self._run(wav, seed)
        out = torch.zeros_like(wav)
        for k, s in enumerate(range(0, wav.shape[-1], self.chunk)):
            seg = wav[..., s:s + self.chunk]
            out[..., s:s + seg.shape[-1]] = self._run(seg, seed + k)
        return out


def load_diffwave(ckpt_path, t=3, device="cuda", chunk=None):
    return DiffWavePurifier(ckpt_path, t=t, device=device, chunk=chunk)
