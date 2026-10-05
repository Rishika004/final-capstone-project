"""
test_vocalbridge_proxy.py
=========================
Simulates VocalBridge's core attack mechanism using EnCodec as a proxy.

VocalBridge works in EnCodec latent space:
  protected audio → EnCodec encoder → latent → purify → EnCodec decoder → purified audio

Since VocalBridge code is not public yet, we simulate its bottleneck:
  protected audio → EnCodec encode → decode (latent roundtrip)

If your protection survives this, it has a good chance against VocalBridge.
Then we measure cosine similarity with WavLM to see if cloning still fails.

Requirements:
  pip install encodec transformers torch torchaudio soundfile
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
import os, sys

ORIGINAL_WAV  = "data/samples/test.wav"          # change if needed
PROTECTED_WAV = "protected.wav"
OUT_DIR       = "vocalbridge_test"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Step 1: Load audio ────────────────────────────────────────────────────────
def load_wav(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.to(DEVICE)

print("\n[1] Loading audio files...")
try:
    wav_original  = load_wav(ORIGINAL_WAV)
    print(f"    Original  : {ORIGINAL_WAV}  shape={list(wav_original.shape)}")
except Exception as e:
    print(f"    WARNING: Could not load original: {e}")
    wav_original = None

wav_protected = load_wav(PROTECTED_WAV)
print(f"    Protected : {PROTECTED_WAV}  shape={list(wav_protected.shape)}")

# ── Step 2: EnCodec encode-decode (VocalBridge latent bottleneck proxy) ───────
print("\n[2] Running EnCodec encode → decode (VocalBridge proxy)...")
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model.to(DEVICE).eval()

    def encodec_roundtrip(wav, sr=16000):
        # EnCodec expects 24kHz
        wav_24k = torchaudio.functional.resample(wav, sr, 24000)
        wav_24k = wav_24k.unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            encoded = encodec_model.encode(wav_24k)
            decoded = encodec_model.decode(encoded)
        decoded = decoded.squeeze(0)  # [1, T]
        # resample back to 16kHz
        decoded_16k = torchaudio.functional.resample(decoded, 24000, 16000)
        return decoded_16k.clamp(-1.0, 1.0)

    wav_protected_purified = encodec_roundtrip(wav_protected)
    print(f"    Purified shape: {list(wav_protected_purified.shape)}")

    # Save purified audio
    purified_path = os.path.join(OUT_DIR, "protected_after_vocalbridge_proxy.wav")
    torchaudio.save(purified_path, wav_protected_purified.cpu(), 16000)
    print(f"    Saved: {purified_path}")

    if wav_original is not None:
        wav_original_purified = encodec_roundtrip(wav_original)
        orig_purified_path = os.path.join(OUT_DIR, "original_after_vocalbridge_proxy.wav")
        torchaudio.save(orig_purified_path, wav_original_purified.cpu(), 16000)
        print(f"    Saved: {orig_purified_path}")

except ImportError:
    print("    encodec not installed. Installing...")
    os.system(f"{sys.executable} -m pip install encodec -q")
    print("    Please re-run the script after installation.")
    sys.exit(1)
except Exception as e:
    print(f"    EnCodec failed: {e}")
    print("    Falling back to MP3 compression proxy...")

    # Fallback: MP3 compression as alternative purifier proxy
    import io
    def mp3_roundtrip(wav, sr=16000):
        buf = io.BytesIO()
        torchaudio.save(buf, wav.cpu(), sr, format="mp3")
        buf.seek(0)
        wav_mp3, _ = torchaudio.load(buf)
        if wav_mp3.shape[-1] != wav.shape[-1]:
            wav_mp3 = torch.nn.functional.pad(wav_mp3, (0, wav.shape[-1] - wav_mp3.shape[-1]))
        return wav_mp3.to(DEVICE)

    wav_protected_purified = mp3_roundtrip(wav_protected)
    purified_path = os.path.join(OUT_DIR, "protected_after_mp3_proxy.wav")
    torchaudio.save(purified_path, wav_protected_purified.cpu(), 16000)
    print(f"    Saved (MP3 proxy): {purified_path}")

# ── Step 3: WavLM cosine similarity check ────────────────────────────────────
print("\n[3] Checking WavLM cosine similarity (does cloning still fail?)...")
try:
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    print("    Loading WavLM-Base-Plus...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()

    def get_embedding(wav):
        wav_np = wav.squeeze().cpu().numpy()
        inputs = feature_extractor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(DEVICE)
        with torch.no_grad():
            out = wavlm(input_values)
        emb = out.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    emb_protected          = get_embedding(wav_protected)
    emb_protected_purified = get_embedding(wav_protected_purified)

    sim_prot_vs_purified = torch.nn.functional.cosine_similarity(
        emb_protected, emb_protected_purified
    ).item()

    print(f"\n    cos(protected, after_purification) = {sim_prot_vs_purified:.4f}")

    if wav_original is not None:
        emb_original = get_embedding(wav_original)
        sim_orig_vs_purified = torch.nn.functional.cosine_similarity(
            emb_original, emb_protected_purified
        ).item()
        sim_orig_vs_protected = torch.nn.functional.cosine_similarity(
            emb_original, emb_protected
        ).item()

        print(f"    cos(original, protected)           = {sim_orig_vs_protected:.4f}")
        print(f"    cos(original, after_purification)  = {sim_orig_vs_purified:.4f}")

        print("\n" + "="*60)
        print("  VERDICT")
        print("="*60)
        print(f"  Original vs Protected        : {sim_orig_vs_protected:.4f}  {'[CLONING FAILS]' if sim_orig_vs_protected < 0.5 else '[CLONING WORKS]'}")
        print(f"  Original vs After Purification: {sim_orig_vs_purified:.4f}  {'[CLONING FAILS]' if sim_orig_vs_purified < 0.5 else '[CLONING WORKS]'}")
        print("="*60)

        if sim_orig_vs_purified < 0.5:
            print("  RESULT: Protection SURVIVES VocalBridge proxy!")
            print("  Your null-space delta holds even after EnCodec roundtrip.")
        else:
            print("  RESULT: Protection STRIPPED by VocalBridge proxy.")
            print("  Similarity recovered above 0.5 — cloning would succeed.")
            print("  Next step: extend null space to EnCodec latent space.")

except Exception as e:
    print(f"    WavLM check failed: {e}")
    print("    You can manually listen to the purified audio and compare.")

print(f"\n[Done] Output files saved to: {OUT_DIR}/")
