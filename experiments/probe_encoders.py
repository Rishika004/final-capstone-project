"""Probe ECAPA + Resemblyzer for differentiability and module structure."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path: sys.path.insert(0, PROJ)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav = torch.randn(1, 16000, device=device) * 0.05
wav.requires_grad_(True)

print("=" * 60)
print("ECAPA (speechbrain)")
print("=" * 60)
from speechbrain.inference.speaker import EncoderClassifier
SAVE = os.path.join(PROJ, "experiments", "sb_models")
ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(SAVE, "ecapa"), run_opts={"device": str(device)})
print("mods keys:", list(ecapa.mods.keys()))
# Try encode_batch with grad
try:
    emb = ecapa.encode_batch(wav)
    print("encode_batch emb shape:", emb.shape, "requires_grad:", emb.requires_grad)
    if emb.requires_grad:
        emb.sum().backward()
        print("encode_batch grad flows:", wav.grad is not None and wav.grad.abs().sum().item() > 0)
    else:
        print("encode_batch: NO grad (wrapped in no_grad)")
except Exception as e:
    print("encode_batch failed:", e)

# Try manual differentiable path via mods
wav.grad = None
try:
    feats = ecapa.mods.compute_features(wav)          # Fbank
    feats = ecapa.mods.mean_var_norm(feats, torch.ones(1, device=device))
    emb2 = ecapa.mods.embedding_model(feats)
    print("manual mods emb shape:", emb2.shape, "requires_grad:", emb2.requires_grad)
    emb2.sum().backward()
    g = wav.grad.abs().sum().item() if wav.grad is not None else 0.0
    print("manual mods grad flows:", g > 0, "grad_sum=%.4e" % g)
except Exception as e:
    import traceback; traceback.print_exc()

print()
print("=" * 60)
print("Resemblyzer d-vector")
print("=" * 60)
from resemblyzer import VoiceEncoder
from resemblyzer.hparams import (sampling_rate, mel_window_length,
                                 mel_window_step, mel_n_channels)
print(f"sr={sampling_rate} win_len_ms={mel_window_length} "
      f"step_ms={mel_window_step} n_mels={mel_n_channels}")
dvec = VoiceEncoder(device=device)
print("VoiceEncoder modules:", [n for n, _ in dvec.named_children()])
# forward signature: forward(self, mels) mels [batch, n_frames, n_channels]
# Check if we can build torch mels and get grad
import torchaudio.transforms as T
n_fft = int(sampling_rate * mel_window_length / 1000)   # 400
hop = int(sampling_rate * mel_window_step / 1000)        # 160
print(f"derived n_fft={n_fft} hop={hop}")
melspec = T.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft,
                           hop_length=hop, n_mels=mel_n_channels).to(device)
wav.grad = None
try:
    m = melspec(wav)                    # [1, 40, frames]
    m = m.squeeze(0).transpose(0, 1).unsqueeze(0)   # [1, frames, 40]
    emb3 = dvec(m)
    print("dvec(torch mel) emb shape:", emb3.shape, "requires_grad:", emb3.requires_grad)
    emb3.sum().backward()
    g = wav.grad.abs().sum().item() if wav.grad is not None else 0.0
    print("dvec grad flows:", g > 0, "grad_sum=%.4e" % g)
except Exception as e:
    import traceback; traceback.print_exc()
