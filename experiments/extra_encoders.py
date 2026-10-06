"""
experiments/extra_encoders.py
=============================
CAM++ and ERes2Net as **evaluation-only, differentiable** speaker encoders, for
council #13b.

WHY THESE TWO, AND WHY NOT IN THE CRAFTING ENSEMBLE
---------------------------------------------------
C1 says each encoder's own identity direction survives purification worse than
a matched random direction, and that *how much* differs per encoder. §32/§37/§43
measure that on the three encoders the perturbation was crafted against, so the
explanation was generated after seeing the behaviour it explains — `FRAME_v6` §4
is explicit that this must be written as "an explanation generated after
observing the reversal", never as a prediction.

#13b turns it into a prediction instead, for ~2 GPU-h rather than the ~12 that
adding a crafting encoder would cost (§24b's lesson: add new encoders as
eval-only). Two encoders neither the optimiser nor the analysis has ever seen:

  * **CAM++** — a context-aware masking TDNN (3D-Speaker), architecturally
    closest to ECAPA of the two.
  * **ERes2Net** — a Res2Net with attentional feature fusion, a different
    family again from the transformer (WavLM-SV), the channel-attentive CNN
    (ECAPA) and the LSTM (d-vector).

Both are VoxCeleb-trained English models, matching the corpus. They are
**never** added to `build_encoders`: nothing here may change crafting.

WHAT HAS TO BE DIFFERENTIABLE, AND WHAT DOES NOT
------------------------------------------------
α(g_e) needs the gradient of the encoder's own similarity w.r.t. the mel, so
the feature extraction has to be differentiable too. 3D-Speaker uses Kaldi
fbank with cepstral mean normalisation; `torchaudio.compliance.kaldi.fbank` is
written in torch ops and carries gradients, so the whole path does.

Scoring existing wavs through these encoders needs only a forward pass.

PROVENANCE
----------
Model definitions are vendored **byte-identical** from modelscope/3D-Speaker at
`external/3dspeaker/speakerlab/models/` (Apache-2.0), with only `__init__.py`
files added so the upstream intra-package imports resolve. The `speakerlab`
PyPI package was deliberately NOT installed: it pulls funasr, modelscope,
datasets, hdbscan, umap-learn and the Aliyun SDKs, and would downgrade `fsspec`
under a working torch/transformers environment. Checkpoints come from
ModelScope `iic/speech_campplus_sv_en_voxceleb_16k` (29.4 MB) and
`iic/speech_eres2net_sv_en_voxceleb_16k` (26.7 MB).

Run the self-test:
  python experiments/extra_encoders.py
"""
import os
import sys

import torch
import torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(PROJ, "external", "3dspeaker")
CKPT = os.path.join(VENDOR, "ckpt")
SR = 16000


def _ensure_path():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    if not os.path.isdir(os.path.join(VENDOR, "speakerlab")):
        raise SystemExit(
            f"[extra] vendored 3D-Speaker model code not found at {VENDOR}.\n"
            f"  It is gitignored (external/), like NISQA and De-AntiFake. "
            f"Re-fetch speakerlab/models/{{campplus,eres2net}} from "
            f"modelscope/3D-Speaker and add empty __init__.py files.")


def fbank(wav, n_mels=80):
    """80-dim Kaldi fbank with cepstral mean normalisation, differentiable.

    This mirrors 3D-Speaker's own front end. `torchaudio.compliance.kaldi` is
    implemented in torch ops, so the gradient reaches the waveform -- which is
    what α(g_e) needs and what an ONNX export could not give us.
    """
    import torchaudio.compliance.kaldi as K
    w = wav if wav.dim() == 2 else wav.unsqueeze(0)
    # Kaldi's fbank expects the conventional int16-scaled range.
    feat = K.fbank(w * (1 << 15), num_mel_bins=n_mels, sample_frequency=SR,
                   dither=0.0)
    return feat - feat.mean(dim=0, keepdim=True)


class _Base:
    """Common shape contract: .embed(wav[1,T]) -> L2-normalised [1, D]."""

    def embed(self, wav):
        f = fbank(wav).unsqueeze(0)                 # [1, frames, 80]
        e = self.m(f)
        if isinstance(e, (tuple, list)):
            e = e[-1]
        return F.normalize(e.float().reshape(1, -1), dim=-1)


class CAMPPlusEnc(_Base):
    name = "CAM++"
    threshold = 0.0          # placeholder: calibrate before any binary decision

    def __init__(self, device, ckpt=None):
        _ensure_path()
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        p = ckpt or os.path.join(CKPT, "campplus_voxceleb.bin")
        if not os.path.isfile(p):
            raise SystemExit(f"[extra] missing checkpoint {p}")
        self.m = CAMPPlus(feat_dim=80, embedding_size=512)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        self.m.load_state_dict(sd)
        self.m = self.m.to(device).eval()
        for q in self.m.parameters():
            q.requires_grad_(False)


class ERes2NetEnc(_Base):
    name = "ERes2Net"
    threshold = 0.0

    def __init__(self, device, ckpt=None):
        _ensure_path()
        from speakerlab.models.eres2net.ERes2Net import ERes2Net
        p = ckpt or os.path.join(CKPT, "pretrained_eres2net.ckpt")
        if not os.path.isfile(p):
            raise SystemExit(f"[extra] missing checkpoint {p}")
        self.m = ERes2Net(feat_dim=80, embedding_size=192)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        self.m.load_state_dict(sd)
        self.m = self.m.to(device).eval()
        for q in self.m.parameters():
            q.requires_grad_(False)


def build_extra(device):
    """Both encoders, in a fixed order. NEVER added to build_encoders()."""
    return [CAMPPlusEnc(device), ERes2NetEnc(device)]


def _self_test():
    """Load both, and check they behave like speaker encoders at all.

    Two checks, because a model that loads is not a model that works:
      1. same speaker scores higher than different speaker, on real audio;
      2. the gradient w.r.t. the waveform is finite and non-zero, which is the
         property α depends on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from experiments.librispeech import (list_librispeech, reference_items,
                                         cross_session_split, load_flac)
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0, utt_index=0)
    enr = reference_items(items, min_sec=4.0, utt_index=1)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][:6]

    for enc in build_extra(device):
        print(f"\n--- {enc.name} ---")
        embs_a = {s: enc.embed(load_flac(ref[s]["path"], 4.0).to(device))
                  for s in spks}
        embs_b = {s: enc.embed(load_flac(enr[s]["path"], 4.0).to(device))
                  for s in spks if s in enr}
        same, diff = [], []
        for s in embs_b:
            same.append(float((embs_a[s] * embs_b[s]).sum()))
            for t in embs_b:
                if t != s:
                    diff.append(float((embs_a[s] * embs_b[t]).sum()))
        import statistics as st
        print(f"  same-speaker  mean {st.mean(same):+.3f}  (n={len(same)})")
        print(f"  diff-speaker  mean {st.mean(diff):+.3f}  (n={len(diff)})")
        sep = st.mean(same) - st.mean(diff)
        print(f"  separation    {sep:+.3f}  "
              f"{'OK' if sep > 0.2 else 'SUSPECT -- check the checkpoint'}")

        w = load_flac(ref[spks[0]]["path"], 4.0).to(device).requires_grad_(True)
        o = enc.embed(w)
        g, = torch.autograd.grad((o * o.detach()).sum(), w)
        print(f"  grad w.r.t. waveform: finite={bool(torch.isfinite(g).all())} "
              f"norm={float(g.norm()):.4e} "
              f"{'OK' if torch.isfinite(g).all() and g.norm() > 0 else 'BROKEN'}")


if __name__ == "__main__":
    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)
    _self_test()
