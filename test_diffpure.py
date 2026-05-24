"""
test_diffpure.py
================
Tests whether our null-space protection survives real DiffPure.

Run with:
    python test_diffpure.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*50}")
print(f"  DiffPure Resistance Test")
print(f"{'='*50}")
print(f"Device: {device}")

# ---------------------------------------------------------------
# Load models
# ---------------------------------------------------------------
print("\n[1] Loading models...")
from models.purifier import load_pretrained as load_purifier
from models.speaker_encoder import load_pretrained as load_encoder
from models.real_diffpure import load_real_diffpure

surrogate  = load_purifier(device=device)
real_dp    = load_real_diffpure(t_diffuse=100, device=device)
encoder    = load_encoder(device=device)

surrogate.eval()
real_dp.eval()
encoder.eval()

# ---------------------------------------------------------------
# Load audio
# ---------------------------------------------------------------
print("\n[2] Loading audio files...")
from utils.audio_io import load_wav

orig_wav, _ = load_wav('data/samples/sample_001.wav')
prot_wav, _ = load_wav('results_pgd_bypass/sample_001_protected.wav')

# Align lengths
n        = min(orig_wav.shape[-1], prot_wav.shape[-1])
orig_wav = orig_wav[..., :n].to(device)
prot_wav = prot_wav[..., :n].to(device)

print(f"  Audio length : {n / 16000:.2f}s")
print(f"  Waveform shape : {list(orig_wav.shape)}")

# ---------------------------------------------------------------
# Convert to mel
# ---------------------------------------------------------------
mel_fn   = T.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
).to(device)

mel_orig = torch.log(mel_fn(orig_wav) + 1e-9)
mel_prot = torch.log(mel_fn(prot_wav) + 1e-9)

print(f"  Mel shape : {list(mel_orig.shape)}")

# ---------------------------------------------------------------
# Step 1: Speaker similarity BEFORE any purification
# ---------------------------------------------------------------
print(f"\n[3] Measuring speaker similarity BEFORE purification...")
with torch.no_grad():
    emb_o = F.normalize(encoder(orig_wav), dim=-1)
    emb_p = F.normalize(encoder(prot_wav), dim=-1)
    sim_before = (emb_o * emb_p).sum().item()

print(f"  Speaker similarity (orig vs protected) : {sim_before:.4f}")
if sim_before < 0.5:
    print("  --> Cloning FAILS before purification  (good)")
else:
    print("  --> Cloning WORKS before purification  (bad)")

# ---------------------------------------------------------------
# Step 2: Run through SURROGATE purifier (single pass)
# ---------------------------------------------------------------
print(f"\n[4] Applying SURROGATE purifier (single forward pass)...")
with torch.no_grad():
    mel_orig_surr = surrogate(mel_orig)
    mel_prot_surr = surrogate(mel_prot)

    prr_surr = (
        (mel_prot_surr - mel_orig_surr).norm() /
        ((mel_orig_surr - mel_orig).norm() + 1e-8)
    ).item()

print(f"  PRR after surrogate : {prr_surr:.4f}")
if prr_surr >= 1.0:
    print("  --> Protection SURVIVES surrogate purifier  (good)")
else:
    print("  --> Protection weakened by surrogate purifier")

# ---------------------------------------------------------------
# Step 3: Run through REAL DiffPure (full DDPM loop)
# ---------------------------------------------------------------
print(f"\n[5] Applying REAL DiffPure (DDPM loop, t=100 steps)...")
print("    This may take 30-60 seconds...")

with torch.no_grad():
    mel_orig_dp = real_dp(mel_orig, seed=42)
    mel_prot_dp = real_dp(mel_prot, seed=42)

    prr_dp = (
        (mel_prot_dp - mel_orig_dp).norm() /
        ((mel_orig_dp - mel_orig).norm() + 1e-8)
    ).item()

    mel_dist_orig = (mel_orig_dp - mel_orig).abs().mean().item()
    mel_dist_prot = (mel_prot_dp - mel_orig).abs().mean().item()

print(f"  PRR after real DiffPure (t=100) : {prr_dp:.4f}")
print(f"  Mel change on original audio    : {mel_dist_orig:.6f}")
print(f"  Mel change on protected audio   : {mel_dist_prot:.6f}")

# ---------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  FINAL RESULTS")
print(f"{'='*50}")
print(f"  Speaker sim BEFORE purification    : {sim_before:.4f}  (target < 0.5)")
print(f"  PRR after surrogate purifier       : {prr_surr:.4f}  (target > 1.0)")
print(f"  PRR after real DiffPure (t=100)    : {prr_dp:.4f}  (target > 1.0)")
print(f"{'='*50}")

if prr_dp >= 1.0:
    print("  RESULT: Protection SURVIVES real DiffPure attack  ✅")
    print("  --> Null space transfers from surrogate to DDPM loop")
elif prr_dp >= 0.5:
    print("  RESULT: Protection PARTIALLY survives  ⚠️")
    print("  --> Null space partially transfers — needs stronger epsilon")
else:
    print("  RESULT: Protection does NOT survive real DiffPure  ❌")
    print("  --> Need to rethink surrogate or increase t_diffuse")

print(f"{'='*50}\n")
