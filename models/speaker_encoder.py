"""
models/speaker_encoder.py
=========================
Speaker embedding extractor using Microsoft WavLM.

WavLM (Wang et al., 2022) is a self-supervised speech representation model
that captures speaker identity, linguistic content, and prosody.  We use
the "base-plus" variant as a lightweight but high-quality speaker encoder.

The model outputs a 256-dimensional speaker embedding per audio clip.
These embeddings can be compared with cosine similarity to judge whether
two recordings come from the same speaker — which is exactly what voice
cloning systems do.

Our defence goal:  after applying null-space perturbation, the cosine
similarity between the protected audio's embedding and the original
speaker's embedding should drop as far toward 0 as possible.

Model: microsoft/wavlm-base-plus
Input: raw waveform, 16 kHz mono, shape [batch, T]  or [T]
Output: speaker embedding, shape [batch, 256]
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
_EMBED_DIM = 256  # we project the 768-dim WavLM hidden states to 256


# ---------------------------------------------------------------------------
# Speaker Encoder class
# ---------------------------------------------------------------------------

class WavLMSpeakerEncoder(nn.Module):
    """
    Speaker encoder backed by WavLM-Base-Plus.

    Workflow
    --------
    1. Pass raw waveform through WavLM → hidden states [batch, T', 768].
    2. Mean-pool over the time dimension → [batch, 768].
    3. Project to 256-d with a linear layer → [batch, 256].
    4. L2-normalise to unit sphere (so cosine similarity == dot product).

    The projection layer is randomly initialised; only WavLM weights are
    loaded from HuggingFace.  In a production system you would fine-tune
    the projection head on a speaker-verification dataset (e.g. VoxCeleb).
    """

    def __init__(self, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.wavlm = None          # loaded lazily in `load_pretrained`
        self.feature_extractor = None
        # Linear projection: WavLM hidden size (768) → embed_dim (256)
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from raw waveform(s).

        Parameters
        ----------
        waveform : torch.Tensor
            Shape [batch, T] or [T].  Expected sample rate: 16 000 Hz.
            Values should be in [-1, 1] (normalised float).

        Returns
        -------
        embedding : torch.Tensor, shape [batch, embed_dim]
            L2-normalised speaker embedding.
        """
        if self.wavlm is None:
            raise RuntimeError(
                "WavLM not loaded. Call load_pretrained() first or use "
                "WavLMSpeakerEncoder.from_pretrained()."
            )

        # Ensure [batch, T]
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True

        # HuggingFace WavLM expects numpy or tensor on the correct device
        # We call the model directly (bypassing the pipeline) for grad flow
        outputs = self.wavlm(
            input_values=waveform,
            output_hidden_states=False,
        )

        # last_hidden_state: [batch, T', 768]
        hidden = outputs.last_hidden_state

        # Mean-pool over time
        pooled = hidden.mean(dim=1)  # [batch, 768]

        # Project to embed_dim
        embedding = self.projection(pooled)  # [batch, 256]

        # L2 normalise
        embedding = F.normalize(embedding, dim=-1)

        if squeeze:
            embedding = embedding.squeeze(0)

        return embedding


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two speaker embeddings.

    Parameters
    ----------
    emb1, emb2 : torch.Tensor
        Shape [embed_dim] or [batch, embed_dim].

    Returns
    -------
    similarity : torch.Tensor
        Scalar (or [batch]) cosine similarity in [-1, 1].
        1.0 = identical speaker,  0.0 = orthogonal,  -1.0 = opposite.
    """
    emb1 = F.normalize(emb1.float(), dim=-1)
    emb2 = F.normalize(emb2.float(), dim=-1)
    return (emb1 * emb2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Factory / loader
# ---------------------------------------------------------------------------

def load_pretrained(
    device: Optional[torch.device] = None,
    model_id: str = _WAVLM_MODEL_ID,
) -> WavLMSpeakerEncoder:
    """
    Load the WavLM-Base-Plus speaker encoder from HuggingFace.

    Parameters
    ----------
    device : torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.
    model_id : str
        HuggingFace model identifier.

    Returns
    -------
    encoder : WavLMSpeakerEncoder
        Speaker encoder in eval mode.

    Notes
    -----
    On first call this will download ~360 MB of model weights.
    Subsequent calls use the HuggingFace cache.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from transformers import WavLMModel

        logger.info("Loading WavLM from '%s' ...", model_id)
        wavlm = WavLMModel.from_pretrained(model_id)
        wavlm = wavlm.to(device)
        wavlm.eval()

        encoder = WavLMSpeakerEncoder(embed_dim=_EMBED_DIM)
        encoder.wavlm = wavlm
        encoder = encoder.to(device)
        encoder.eval()

        n_params = sum(p.numel() for p in encoder.parameters())
        logger.info(
            "WavLMSpeakerEncoder ready on %s | parameters = %d (%.1f M)",
            device, n_params, n_params / 1e6,
        )
        print(
            f"[SpeakerEncoder] WavLM-Base-Plus loaded on {device} | "
            f"{n_params/1e6:.1f}M params"
        )
        return encoder

    except ImportError:
        logger.error(
            "transformers library not found. Install with: "
            "pip install transformers"
        )
        raise
    except Exception as exc:
        logger.warning(
            "Could not load WavLM from HuggingFace (%s). "
            "Falling back to dummy encoder.",
            exc,
        )
        return _make_dummy_encoder(device)


def _make_dummy_encoder(
    device: torch.device,
    embed_dim: int = _EMBED_DIM,
) -> nn.Module:
    """
    Fallback: random linear speaker encoder (no HuggingFace needed).

    Useful for offline testing or unit tests.  Embeddings are meaningless
    but the pipeline still runs end-to-end.
    """
    logger.warning("Using DUMMY speaker encoder — embeddings are random.")
    print("[SpeakerEncoder] WARNING: using dummy random encoder.")

    class DummySpeakerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Project from raw samples — just to have some parameters
            self.proj = nn.Linear(1, embed_dim)

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            # Mean over samples → [batch, 1] → project → [batch, embed_dim]
            mean_feat = waveform.mean(dim=-1, keepdim=True)
            emb = self.proj(mean_feat)
            return F.normalize(emb, dim=-1)

    enc = DummySpeakerEncoder().to(device)
    enc.eval()
    return enc
