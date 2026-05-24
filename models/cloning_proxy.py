"""
models/cloning_proxy.py
=======================
Surrogate voice cloning model — a differentiable proxy for real TTS/VC systems.

Why a proxy?
------------
Real voice cloning systems (VALL-E, YourTTS, FreeVC, etc.) are:
  - Large (1–5 GB), slow to run, and require careful setup.
  - Not differentiable end-to-end (they use autoregressive sampling).

For our adversarial training we only need a differentiable model that
simulates the speaker-embedding extraction step of a cloning pipeline,
since that is the part we want to fool.

Design
------
The cloning proxy is a two-stage model:
  1. Encoder  — extracts a speaker embedding from the input waveform.
                (Backed by WavLM, same as speaker_encoder.py.)
  2. Synthesizer stub — a shallow MLP that maps (content_embedding,
                        speaker_embedding) → output_mel.
                        This simulates the "clone" step.

For the null-space perturbation objective we only care about whether
the speaker embedding of the cloned output matches the target speaker.
Therefore `cloning_success_score` returns the cosine similarity between:
  - The cloned output's speaker embedding
  - The original (target) speaker's embedding

A HIGH score means the clone succeeded (bad for privacy).
A LOW score means the clone failed   (good for our defence).

Usage
-----
    proxy  = CloningProxy()
    proxy.load_pretrained()

    # Does the protected audio fool the cloner?
    score = proxy.cloning_success_score(wav_protected, wav_original)
    # score ≈ 0.0 → defence succeeded; score ≈ 1.0 → clone still works
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shallow synthesizer stub
# ---------------------------------------------------------------------------

class SynthesizerStub(nn.Module):
    """
    Minimal MLP that pretends to synthesise a mel spectrogram from
    content + speaker representations.

    Inputs
    ------
    content_emb : [batch, content_dim]
    speaker_emb : [batch, speaker_dim]

    Output
    ------
    mel_out : [batch, n_mels, time_steps]  (flattened and reshaped)
    """

    def __init__(
        self,
        content_dim: int = 256,
        speaker_dim: int = 256,
        n_mels: int = 80,
        time_steps: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.time_steps = time_steps
        out_dim = n_mels * time_steps

        self.net = nn.Sequential(
            nn.Linear(content_dim + speaker_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        content_emb: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([content_emb, speaker_emb], dim=-1)
        out_flat = self.net(x)
        return out_flat.reshape(-1, self.n_mels, self.time_steps)


# ---------------------------------------------------------------------------
# Full cloning proxy
# ---------------------------------------------------------------------------

class CloningProxy(nn.Module):
    """
    Differentiable surrogate for a voice cloning pipeline.

    Pipeline (simplified):
        wav_source → content_encoder → content_emb  ┐
        wav_target → speaker_encoder → speaker_emb  ┤→ synthesizer → mel_clone
                                                     └→ (for eval) re-encode mel_clone
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_mels: int = 80,
        time_steps: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Content encoder: same architecture as speaker encoder (shares backbone
        # in real systems; here we keep it independent for simplicity)
        self._content_encoder: Optional[nn.Module] = None
        self._speaker_encoder: Optional[nn.Module] = None

        self.synthesizer = SynthesizerStub(
            content_dim=embed_dim,
            speaker_dim=embed_dim,
            n_mels=n_mels,
            time_steps=time_steps,
        )

    # ------------------------------------------------------------------
    # Core forward: clone source voice with target speaker identity
    # ------------------------------------------------------------------

    def forward(
        self,
        wav_source: torch.Tensor,
        wav_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clone wav_source using the speaker identity from wav_target.

        Parameters
        ----------
        wav_source : torch.Tensor, shape [batch, T]
            Audio whose *content* (words/phonemes) we want to clone.
        wav_target : torch.Tensor, shape [batch, T]
            Audio whose *speaker identity* we want to clone.

        Returns
        -------
        mel_clone : torch.Tensor, shape [batch, n_mels, time_steps]
            Synthesised mel spectrogram with source content + target voice.
        """
        content_emb = self._encode(wav_source)   # [batch, embed_dim]
        speaker_emb = self._encode(wav_target)    # [batch, embed_dim]
        mel_clone = self.synthesizer(content_emb, speaker_emb)
        return mel_clone

    # ------------------------------------------------------------------
    # Defence evaluation helper
    # ------------------------------------------------------------------

    def cloning_success_score(
        self,
        wav_protected: torch.Tensor,
        wav_original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Measure how well the cloner can reproduce the original speaker's
        identity from the protected audio.

        Returns cosine similarity ∈ [-1, 1]:
          - Score ≈  1.0 → clone succeeded (bad for privacy).
          - Score ≈  0.0 → clone failed   (good for our defence).

        Parameters
        ----------
        wav_protected : torch.Tensor, shape [batch, T]
            Protected (perturbed) audio.
        wav_original : torch.Tensor, shape [batch, T]
            Original (unprotected) audio, used as the target identity.

        Returns
        -------
        score : torch.Tensor, scalar
        """
        # Attempt cloning: use protected audio as source, original as target
        mel_cloned = self.forward(wav_protected, wav_original)

        # Re-encode the cloned output to get its speaker embedding
        # We flatten the mel to a "waveform-like" 1-D tensor as input to encoder
        batch = mel_cloned.shape[0]
        mel_flat = mel_cloned.reshape(batch, -1)  # [batch, n_mels*time]
        emb_clone = F.normalize(self._encode_flat(mel_flat), dim=-1)

        # Original speaker embedding
        emb_original = F.normalize(self._encode(wav_original), dim=-1)

        score = (emb_clone * emb_original).sum(dim=-1).mean()
        return score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode waveform to embedding using the speaker encoder."""
        if self._speaker_encoder is None:
            raise RuntimeError("Call load_pretrained() before using CloningProxy.")
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return self._speaker_encoder(wav)

    def _encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a flat feature tensor (mel flatten) via a linear projection."""
        if not hasattr(self, '_flat_proj'):
            in_dim = x.shape[-1]
            self._flat_proj = nn.Linear(in_dim, self.embed_dim).to(x.device)
        return self._flat_proj(x)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    def load_pretrained(
        self,
        device: Optional[torch.device] = None,
    ) -> "CloningProxy":
        """
        Load speaker encoder backend.

        Parameters
        ----------
        device : torch.device, optional

        Returns
        -------
        self
        """
        from models.speaker_encoder import load_pretrained as load_wavlm

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._speaker_encoder = load_wavlm(device=device)
        self = self.to(device)
        self.eval()

        logger.info("CloningProxy loaded on %s", device)
        print(f"[CloningProxy] Ready on {device}")
        return self


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_pretrained(
    device: Optional[torch.device] = None,
) -> CloningProxy:
    """
    Convenience factory: build and load a CloningProxy.

    Parameters
    ----------
    device : torch.device, optional

    Returns
    -------
    proxy : CloningProxy
    """
    proxy = CloningProxy()
    proxy.load_pretrained(device=device)
    return proxy
