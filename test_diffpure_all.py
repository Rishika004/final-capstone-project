"""
test_diffpure_all.py
====================
Tests null-space voice protection using REAL DiffPure everywhere.

- Null space computed through actual DDPM loop (not toy UNet)
- Protection tested against mild (t=50) and aggressive (t=100) DiffPure
- Speaker similarity measured BEFORE and AFTER purification
- Full comparison table printed at the end

Run with:
    python test_diffpure_all.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*65)
print("  NULL-SPACE PROTECTION — Real DiffPure Evaluation")
print("="*65)
print(f"  Device: {device}")

# ---------------------------------------------------------------
# Load models
# ---------------------------------------------------------------
print("\n[1] Loading models ...")
from models.real_diffpure import load_real_diffpure, MelDiffPure
from models.speaker_encoder import load_pretrained as load_encoder
from utils.audio_io import load_wav
from utils.vocoder import invert_mel

# DiffPure with two aggressiveness levels
real_dp_50  = load_real_diffpure(t_diffuse=50,  device=device)  # mild
real_dp_100 = load_real_diffpure(t_diffuse=100, device=device)  # aggressive
encoder     = load_encoder(device=device)

real_dp_50.eval()
real_dp_100.eval()
encoder.eval()

mel_fn = T.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
).to(device)

# ---------------------------------------------------------------
# Helper — compute PRR
# ---------------------------------------------------------------
def compute_prr(mel_protected, mel_original, dp_model, seed=42):
    with torch.no_grad():
        mel_orig_pur = dp_model(mel_original, seed=seed)
        mel_prot_pur = dp_model(mel_protected, seed=seed)
        num = (mel_prot_pur - mel_orig_pur).norm()
        den = (mel_orig_pur - mel_original).norm().clamp(min=1e-8)
        return (num / den).item()

# ---------------------------------------------------------------
# Helper — speaker similarity after DiffPure
# ---------------------------------------------------------------
def sim_after_dp(mel_protected, wav_original, dp_model, seed=42):
    with torch.no_grad():
        mel_purified = dp_model(mel_protected, seed=seed)
        wav_purified = invert_mel(mel_purified.cpu(), method='griffin_lim')
        wav_purified = wav_purified.to(device)
        n    = min(wav_original.shape[-1], wav_purified.shape[-1])
        e_o  = F.normalize(encoder(wav_original[..., :n]),  dim=-1)
        e_p  = F.normalize(encoder(wav_purified[..., :n]),  dim=-1)
        return (e_o * e_p).sum().item()

# ---------------------------------------------------------------
# Helper — protect audio using real DiffPure as surrogate
# ---------------------------------------------------------------
def protect(wav, epsilon, label):
    from core.perturbation import apply_null_space_protection
    mel = torch.log(mel_fn(wav) + 1e-9)
    print(f"\n  [{label}] Generating protection ...")
    print(f"  Null space computed through real DiffPure DDPM loop (t=50)")
    mel_protected, wav_delta = apply_null_space_protection(
        x_mel           = mel,
        purifier        = real_dp_50,     # REAL DiffPure as surrogate
        epsilon         = epsilon,
        K               = 50,             # 50 probes
        svd_threshold   = 1e-3,
        verify          = True,
        mode            = 'pgd',
        x_wav           = wav,
        speaker_encoder = encoder,
        pgd_steps       = 50,
    )
    # Build protected waveform — bypass Griffin-Lim
    if wav_delta is not None:
        n   = wav.shape[-1]
        d   = wav_delta.to(device)
        d   = d[..., :n] if d.shape[-1] > n else F.pad(d, (0, n - d.shape[-1]))
        wav_prot = (wav + d).clamp(-1.0, 1.0)
    else:
        wav_prot = wav.clone()
    # IMPORTANT: use the null-space projected mel (from apply_null_space_protection),
    # NOT the mel recomputed from the waveform bypass.
    # - mel_protected: lives in DiffPure's null space → survives purification → use for PRR/DiffPure tests
    # - wav_prot:      waveform bypass  → directly confuses WavLM → use for before-purification speaker_sim
    return mel_protected, wav_prot

# ---------------------------------------------------------------
# Load original audio
# ---------------------------------------------------------------
print("\n[2] Loading audio ...")
orig_wav, _ = load_wav('data/samples/sample_001.wav')
orig_wav    = orig_wav.to(device)
mel_orig    = torch.log(mel_fn(orig_wav) + 1e-9)
print(f"  Duration : {orig_wav.shape[-1]/16000:.2f}s")
print(f"  Mel shape: {list(mel_orig.shape)}")

# ---------------------------------------------------------------
# Run protection at two epsilon levels
# ---------------------------------------------------------------
print("\n[3] Generating protected audio ...")

mel_p01, wav_p01 = protect(orig_wav, epsilon=0.01, label="eps=0.01")
mel_p05, wav_p05 = protect(orig_wav, epsilon=0.05, label="eps=0.05")

# ---------------------------------------------------------------
# Speaker similarity BEFORE purification
# ---------------------------------------------------------------
print("\n[4] Speaker similarity BEFORE purification ...")
with torch.no_grad():
    e_orig = F.normalize(encoder(orig_wav), dim=-1)

    e_p01  = F.normalize(encoder(wav_p01[..., :orig_wav.shape[-1]]), dim=-1)
    e_p05  = F.normalize(encoder(wav_p05[..., :orig_wav.shape[-1]]), dim=-1)

    sim_p01 = (e_orig * e_p01).sum().item()
    sim_p05 = (e_orig * e_p05).sum().item()

print(f"  eps=0.01 : speaker_sim = {sim_p01:+.4f}")
print(f"  eps=0.05 : speaker_sim = {sim_p05:+.4f}")

# ---------------------------------------------------------------
# PRR — does protection survive DiffPure?
# ---------------------------------------------------------------
print("\n[5] Computing PRR (Purification Resistance Rate) ...")
print("    Running DiffPure t=50  on both versions ...")
prr_01_50  = compute_prr(mel_p01, mel_orig, real_dp_50)
prr_05_50  = compute_prr(mel_p05, mel_orig, real_dp_50)

print("    Running DiffPure t=100 on both versions ...")
prr_01_100 = compute_prr(mel_p01, mel_orig, real_dp_100)
prr_05_100 = compute_prr(mel_p05, mel_orig, real_dp_100)

print(f"  eps=0.01 | PRR(t=50)={prr_01_50:.4f} | PRR(t=100)={prr_01_100:.4f}")
print(f"  eps=0.05 | PRR(t=50)={prr_05_50:.4f} | PRR(t=100)={prr_05_100:.4f}")

# ---------------------------------------------------------------
# Baseline: speaker similarity of ORIGINAL (unprotected) audio
#           after DiffPure — the reference point for the table
# ---------------------------------------------------------------
print("\n[6a] Baseline: speaker_sim of ORIGINAL audio after DiffPure ...")
print("     (This is the floor — DiffPure itself alters the audio)")
sim_orig_after_50  = sim_after_dp(mel_orig, orig_wav, real_dp_50)
sim_orig_after_100 = sim_after_dp(mel_orig, orig_wav, real_dp_100)
print(f"  Original audio | after DiffPure t=50  : {sim_orig_after_50:+.4f}")
print(f"  Original audio | after DiffPure t=100 : {sim_orig_after_100:+.4f}")

# ---------------------------------------------------------------
# Speaker similarity AFTER DiffPure — the critical test
# ---------------------------------------------------------------
print("\n[6b] Speaker similarity AFTER DiffPure purification ...")
print("     (This is what the attacker's cloning system actually sees)")
print("     Running ...")

sim_01_after_50  = sim_after_dp(mel_p01, orig_wav, real_dp_50)
sim_05_after_50  = sim_after_dp(mel_p05, orig_wav, real_dp_50)
sim_01_after_100 = sim_after_dp(mel_p01, orig_wav, real_dp_100)
sim_05_after_100 = sim_after_dp(mel_p05, orig_wav, real_dp_100)

# ---------------------------------------------------------------
# Final results table
# ---------------------------------------------------------------
def verdict(sim):
    return "CLONING FAILS" if sim < 0.5 else "CLONING WORKS"

def prr_label(prr):
    if prr >= 1.0: return "FULLY SURVIVES"
    if prr >= 0.5: return "PARTIALLY SURVIVES"
    return "FAILS"

print("\n" + "="*65)
print("  FINAL RESULTS — Real DiffPure as Surrogate")
print("="*65)
print(f"\n  Surrogate used for null space : Real DiffPure (DDPM t=50)")
print(f"  Jacobian probes               : K=50")
print(f"  PGD steps                     : 50")

print(f"\n  {'Metric':<45} baseline     eps=0.01     eps=0.05")
print("  " + "-"*73)
print(f"  {'Speaker sim BEFORE purification':<45} {'n/a':>8}    {sim_p01:+.4f}      {sim_p05:+.4f}")
print(f"  {'PRR vs DiffPure t=50  (mild)':<45} {'n/a':>8}    {prr_01_50:.4f}      {prr_05_50:.4f}")
print(f"  {'PRR vs DiffPure t=100 (aggressive)':<45} {'n/a':>8}    {prr_01_100:.4f}      {prr_05_100:.4f}")
print(f"  {'Speaker sim AFTER DiffPure t=50  [BASELINE]':<45} {sim_orig_after_50:+.4f}    {sim_01_after_50:+.4f}      {sim_05_after_50:+.4f}")
print(f"  {'Speaker sim AFTER DiffPure t=100 [BASELINE]':<45} {sim_orig_after_100:+.4f}    {sim_01_after_100:+.4f}      {sim_05_after_100:+.4f}")

print("\n  " + "-"*73)
print("  VERDICTS (< 0.5 = CLONING FAILS):")
print(f"  Original (no protection) after DiffPure t=50  : {verdict(sim_orig_after_50)}")
print(f"  eps=0.01 after DiffPure t=50  : {verdict(sim_01_after_50)}")
print(f"  eps=0.01 after DiffPure t=100 : {verdict(sim_01_after_100)}")
print(f"  eps=0.05 after DiffPure t=50  : {verdict(sim_05_after_50)}")
print(f"  eps=0.05 after DiffPure t=100 : {verdict(sim_05_after_100)}")

print("\n  " + "-"*73)
print("  INTERPRETATION:")
print("  speaker_sim < 0.5  -> cloning fails (protection works)")
print("  speaker_sim [baseline] = sim of unprotected audio after DiffPure")
print("    -> shows DiffPure's own reconstruction loss, not caused by us")
print("  PRR >= 1.0         -> perturbation fully survives DiffPure")
print("  PRR >= 0.5         -> perturbation partially survives")
print("="*65 + "\n")
