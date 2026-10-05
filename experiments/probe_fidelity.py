"""Check the differentiable d-vector proxy matches the real resemblyzer path."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path: sys.path.insert(0, PROJ)
import torch, torch.nn.functional as F
from pathlib import Path
from evaluation.evaluate import load_wav
from models.speaker_encoder import load_pretrained as load_wavlm
from experiments.multi_encoders import build_all, real_embed_dvec, DvecEnc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavlm = load_wavlm(device=device)
encs = build_all(device, wavlm, os.path.join(PROJ, "experiments", "sb_models"))
dvec_enc = [e for e in encs if isinstance(e, DvecEnc)][0]

def cossim(a, b):
    a = F.normalize(a.float().reshape(1, -1), dim=-1)
    b = F.normalize(b.float().reshape(1, -1), dim=-1)
    return (a * b).sum().item()

print("proxy-vs-real d-vector cosine on clean audio (want > 0.9):")
for wp in sorted(Path(os.path.join(PROJ, "data", "samples")).glob("*.wav"))[:3]:
    wav = load_wav(str(wp)).to(device)
    with torch.no_grad():
        proxy = dvec_enc.embed(wav).cpu()
    real = real_embed_dvec(dvec_enc.dvec, wav)
    print(f"  {wp.name}: {cossim(proxy, real):.4f}")
