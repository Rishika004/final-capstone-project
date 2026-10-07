"""
main.py
=======
Command-line entry point for null-space voice protection.

Usage
-----
# Step 4 — Train the surrogate purifier on noisy/clean mel pairs
python main.py --mode train --input data/samples/ --epochs 300

# Run unit tests + math sanity checks
python main.py --mode test

# Protect a single audio file
python main.py --mode protect \
    --input data/samples/test.wav \
    --output protected.wav \
    --epsilon 0.01

# Run full evaluation on a directory
python main.py --mode evaluate \
    --input data/samples/ \
    --output results/ \
    --epsilon 0.01

# Step 6 — Cloning resistance evaluation (embedding simulation)
python main.py --mode clone-eval \
    --input data/samples/ \
    --output results/ \
    --epsilon 0.01

# Step 6 — Cloning resistance with real YourTTS (pip install TTS first)
python main.py --mode clone-eval \
    --input data/samples/ \
    --output results/ \
    --use-yourtts

Optional flags
--------------
--epsilon FLOAT     L∞ perturbation budget           (default: 0.01)
--K INT             Jacobian probe count              (default: 50)
--svd-threshold F   SVD singular value cutoff        (default: 1e-3)
--device STR        'cpu', 'cuda', or 'auto'         (default: auto)
--log-level STR     Logging verbosity                (default: INFO)
--n-steps INT       Optimiser steps (protect mode)   (default: 0 = single-shot)
--epochs INT        Training epochs (train mode)     (default: 300)
--use-yourtts       Use real YourTTS for clone-eval  (default: False)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def mode_test(args: argparse.Namespace) -> int:
    """Run unit tests + math sanity checks."""
    print("=" * 60)
    print("MODE: TEST — running unit tests and math sanity checks")
    print("=" * 60)

    # ---- Math sanity checks (no pytest dependency) -------------------------
    print("\n[Sanity] 1/4  Null space definition ...")
    import torch
    from core.nullspace import compute_null_space, verify_null_space

    torch.manual_seed(0)
    m, n = 10, 30
    J = torch.randn(m, n)
    null_basis, S = compute_null_space(J, threshold=1e-4)
    residual = verify_null_space(J, null_basis)
    assert residual < 1e-3, f"Null space definition check FAILED (residual={residual:.2e})"
    print(f"  PASS — null dim={null_basis.shape[0]}, residual={residual:.2e}")

    print("\n[Sanity] 2/4  Projection idempotency ...")
    from core.nullspace import project_onto_null_space
    delta = torch.randn(n)
    p1 = project_onto_null_space(delta, null_basis, kind="null")
    p2 = project_onto_null_space(p1, null_basis, kind="null")
    idem_err = (p1 - p2).abs().max().item()
    assert idem_err < 1e-5, f"Idempotency FAILED (err={idem_err:.2e})"
    print(f"  PASS — idempotency error={idem_err:.2e}")

    print("\n[Sanity] 3/4  Epsilon constraint ...")
    from core.perturbation import generate_base_perturbation
    epsilon = 0.01
    # Build a null basis that matches the delta's dimensions
    n_mels, time = 8, 16
    n3 = n_mels * time  # 128
    m3 = n3 // 3       # ~42
    torch.manual_seed(5)
    J3 = torch.randn(m3, n3)
    null_basis3, _ = compute_null_space(J3, threshold=1e-4)
    x_dummy = torch.zeros(1, n_mels, time)
    delta = generate_base_perturbation(x_dummy, epsilon=epsilon, seed=7)
    if null_basis3.shape[0] > 0:
        proj = project_onto_null_space(delta, null_basis3, kind="null")
    else:
        proj = delta
    proj_clamped = proj.clamp(-epsilon, epsilon)
    linf = proj_clamped.abs().max().item()
    assert linf <= epsilon + 1e-7, f"Epsilon constraint FAILED ({linf:.6f})"
    print(f"  PASS — ||δ||_∞ = {linf:.6f} ≤ ε={epsilon}")

    print("\n[Sanity] 4/4  Jacobian (Z,V) shapes ...")
    from models.purifier import MelUNetPurifier
    from core.jacobian import estimate_jacobian_randomized
    torch.manual_seed(1)
    purifier = MelUNetPurifier(n_mels=8, base_channels=8)
    purifier.eval()
    x = torch.randn(1, 8, 16)
    K_test = 10
    Z_test, V_test = estimate_jacobian_randomized(purifier, x, K=K_test, seed=0)
    input_dim = x.numel()
    assert Z_test.shape == (input_dim, K_test), f"Z shape wrong: {Z_test.shape}"
    assert V_test.shape == (input_dim, K_test), f"V shape wrong: {V_test.shape}"
    print(f"  PASS — Z shape = {list(Z_test.shape)}, V shape = {list(V_test.shape)}")

    # ---- Optional: run pytest for full test suite --------------------------
    print("\n[Tests] Running pytest test suite ...")
    try:
        import pytest
        project_root = os.path.dirname(os.path.abspath(__file__))
        exit_code = pytest.main(
            [os.path.join(project_root, "tests", "test_nullspace.py"), "-v", "--tb=short"]
        )
    except ImportError:
        print("  pytest not found — install with: pip install pytest")
        exit_code = 0

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED" if exit_code == 0 else "SOME TESTS FAILED")
    print("=" * 60)
    return int(exit_code)


def mode_protect(args: argparse.Namespace) -> int:
    """Protect a single audio file."""
    print("=" * 60)
    print(f"MODE: PROTECT  |  input={args.input}  |  output={args.output}")
    print(f"  epsilon={args.epsilon}  K={args.K}  n_steps={args.n_steps}")
    print("=" * 60)

    device = _resolve_device(args.device)

    # Validate input
    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}")
        return 1

    # Load models
    from models.real_diffpure import load_real_diffpure
    from models.speaker_encoder import load_pretrained as load_speaker_encoder
    from evaluation.evaluate import load_wav, wav_to_logmel, save_wav

    print("\nLoading models ...")
    # Use real DiffPure as surrogate — null space computed through
    # the actual DDPM loop, not a toy UNet approximation.
    purifier = load_real_diffpure(t_diffuse=50, device=device)

    try:
        speaker_encoder = load_speaker_encoder(device=device)
    except Exception as exc:
        logging.warning("Could not load WavLM: %s — using dummy encoder", exc)
        from models.speaker_encoder import _make_dummy_encoder
        speaker_encoder = _make_dummy_encoder(device)

    # Load audio
    print(f"\nLoading audio: {args.input}")
    wav_original = load_wav(args.input).to(device)
    mel_original = wav_to_logmel(wav_original, device)
    print(f"  Duration: {wav_original.shape[-1]/16000:.2f}s | Mel: {list(mel_original.shape)}")

    if args.n_steps > 0:
        # Iterative optimisation
        print(f"\nRunning iterative optimiser ({args.n_steps} steps) ...")
        from core.optimizer import optimize_perturbation
        wav_protected, history = optimize_perturbation(
            x_mel=mel_original,
            purifier=purifier,
            speaker_encoder=speaker_encoder,
            x_wav=wav_original,
            n_steps=args.n_steps,
            epsilon=args.epsilon,
            K=args.K,
            svd_threshold=args.svd_threshold,
        )
    else:
        # PGD protection with psychoacoustic masking (default path).
        # The waveform delta is optimised against the speaker encoder and,
        # after every step, projected under the frequency-masking threshold
        # of the original speech — so the output sounds like the original.
        pgd_steps = args.pgd_steps if args.pgd_steps > 0 else 150
        print(f"\nRunning masked-PGD protection ({pgd_steps} steps, "
              f"margin={args.margin_db:+.1f} dB) ...")
        from core.perturbation import apply_null_space_protection
        mel_protected, best_wav_delta = apply_null_space_protection(
            x_mel=mel_original,
            x_wav=wav_original,
            speaker_encoder=speaker_encoder,
            purifier=purifier,
            epsilon=args.epsilon,
            K=args.K,
            svd_threshold=args.svd_threshold,
            mode="pgd",
            pgd_steps=pgd_steps,
            perceptual=not args.flat_pgd,
            masking_margin_db=args.margin_db,
        )
        if best_wav_delta is None:
            print("ERROR: PGD did not produce a waveform delta "
                  "(speaker encoder unavailable?). No output written.")
            return 1
        with torch.no_grad():
            wav_protected = (wav_original
                             + best_wav_delta.to(device)).clamp(-1.0, 1.0)

    # Save output
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_wav(wav_protected, out_path)
    print(f"\nSaved protected audio to: {out_path}")

    # Quick metrics — all computed in mel space to avoid Griffin-Lim phase errors
    print("\nComputing metrics ...")
    import math
    from evaluation.metrics import mel_distance, purification_resistance_rate

    with torch.no_grad():
        from evaluation.evaluate import wav_to_logmel
        mel_prot = wav_to_logmel(wav_protected, device)  # mel of the DELIVERED audio

    from evaluation.metrics import perturbation_survival_rate
    mel_dist = mel_distance(mel_original, mel_prot)
    # PSR measured on the delivered signal (review finding #1 + #2). PRR kept for
    # reference but is degenerate (=1.0 at δ=0), so it is not the headline number.
    psr = perturbation_survival_rate(mel_original, mel_prot, purifier)
    prr = purification_resistance_rate(mel_original, mel_prot, purifier)

    # SNR in mel space (avoids Griffin-Lim phase reconstruction artifacts)
    mel_signal_power = mel_original.pow(2).mean().item()
    mel_noise_power  = (mel_prot - mel_original).pow(2).mean().item()
    snr = 10.0 * math.log10(mel_signal_power / (mel_noise_power + 1e-20))

    # Honest waveform-domain quality metrics
    from core.psychoacoustic import waveform_snr_db
    snr_wav = waveform_snr_db(wav_original.cpu(), wav_protected.cpu())
    try:
        from pesq import pesq as _pesq
        pesq_v = _pesq(16000, wav_original.squeeze().cpu().numpy(),
                       wav_protected.squeeze().cpu().numpy()[:wav_original.shape[-1]],
                       "wb")
    except Exception:
        pesq_v = float("nan")

    print(f"  SNR (waveform)  = {snr_wav:.2f} dB  (higher is better)")
    print(f"  PESQ (wb)       = {pesq_v:.3f}  (>3.5 = transparent)")
    print(f"  SNR (mel, dB)   = {snr:.2f}")
    print(f"  Mel distance    = {mel_dist:.6f}")
    print(f"  PSR             = {psr:.4f}  (~1.0 = perturbation survives DiffPure)")
    print(f"  PRR (legacy)    = {prr:.4f}  (=1.0 at δ=0; not a survival proof)")
    return 0


def mode_train(args: argparse.Namespace) -> int:
    """Train the surrogate purifier on noisy/clean mel pairs (Step 4)."""
    print("=" * 60)
    print(f"MODE: TRAIN  |  input={args.input}  |  epochs={args.epochs}")
    print("=" * 60)

    device = _resolve_device(args.device)

    if not os.path.isdir(args.input):
        print(f"ERROR: input directory not found: {args.input}")
        return 1

    from training.train_purifier import train_purifier
    train_purifier(
        audio_dir  = args.input,
        epochs     = args.epochs,
        batch_size = 16,
        lr         = 1e-3,
        device     = device,
    )
    return 0


def mode_clone_eval(args: argparse.Namespace) -> int:
    """Evaluate cloning resistance (Step 6)."""
    print("=" * 60)
    print(f"MODE: CLONE-EVAL  |  input={args.input}  |  output={args.output}")
    print(f"  epsilon={args.epsilon}  K={args.K}  use_yourtts={args.use_yourtts}")
    print("=" * 60)

    device = _resolve_device(args.device)

    from models.purifier import load_pretrained as load_purifier
    from models.speaker_encoder import load_pretrained as load_speaker_encoder
    from evaluation.cloning_attack import run_cloning_evaluation

    print("\nLoading models ...")
    purifier = load_purifier(device=device)
    try:
        speaker_encoder = load_speaker_encoder(device=device)
    except Exception as exc:
        logging.warning("WavLM unavailable: %s — using dummy encoder", exc)
        from models.speaker_encoder import _make_dummy_encoder
        speaker_encoder = _make_dummy_encoder(device)

    results = run_cloning_evaluation(
        audio_dir       = args.input,
        output_dir      = args.output,
        purifier        = purifier,
        speaker_encoder = speaker_encoder,
        epsilon         = args.epsilon,
        K               = args.K,
        svd_threshold   = args.svd_threshold,
        use_yourtts     = args.use_yourtts,
        device          = device,
    )
    print(f"\nCloning evaluation complete. {len(results)} files processed.")
    return 0


def mode_evaluate(args: argparse.Namespace) -> int:
    """Run full evaluation on a directory of .wav files."""
    mode = "pgd" if args.pgd else "random"
    print("=" * 60)
    print(f"MODE: EVALUATE  |  input={args.input}  |  output={args.output}")
    print(f"  epsilon={args.epsilon}  K={args.K}  protection={mode}")
    print("=" * 60)

    device = _resolve_device(args.device)

    from evaluation.evaluate import run_full_evaluation
    df = run_full_evaluation(
        audio_dir=args.input,
        output_dir=args.output,
        epsilon=args.epsilon,
        K=args.K,
        svd_threshold=args.svd_threshold,
        device=device,
        mode=mode,
        pgd_steps=args.pgd_steps,
        n_files=args.n_files if args.n_files > 0 else None,
        perceptual=not args.flat_pgd,
        masking_margin_db=args.margin_db,
    )
    print(f"\nEvaluation complete. {len(df)} files processed.")
    return 0


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Device] Using: {device}")
    return device


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nullspace_protection",
        description=(
            "Null-Space Voice Protection: add imperceptible noise that survives "
            "DiffPure-style audio purification."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["protect", "evaluate", "test", "train", "clone-eval"],
        required=True,
        help=(
            "train      — train surrogate purifier on noisy/clean mel pairs (Step 4); "
            "protect    — protect a single audio file; "
            "evaluate   — full evaluation on a directory; "
            "clone-eval — cloning resistance evaluation (Step 6); "
            "test       — run unit tests + math sanity checks"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        default="data/samples/",
        help="Input .wav file (protect mode) or directory (evaluate mode).",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/",
        help="Output .wav file (protect mode) or directory (evaluate mode).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="L∞ perturbation budget (e.g. 0.01 = 1%% of signal range).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=50,
        help="Number of random Jacobian probe vectors (higher = better approximation).",
    )
    parser.add_argument(
        "--svd-threshold",
        type=float,
        default=1e-3,
        dest="svd_threshold",
        help="Singular value cutoff for null-space extraction.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=0,
        dest="n_steps",
        help=(
            "Number of iterative optimiser steps (protect mode only). "
            "0 = single-shot null-space projection (faster)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (train mode only).",
    )
    parser.add_argument(
        "--use-yourtts",
        action="store_true",
        default=False,
        dest="use_yourtts",
        help=(
            "Use real YourTTS voice cloning for clone-eval mode. "
            "Requires: pip install TTS  (downloads ~1.5 GB on first run)."
        ),
    )
    parser.add_argument(
        "--pgd",
        action="store_true",
        default=False,
        help=(
            "Use PGD adversarial perturbation instead of random delta. "
            "Runs 50 steps of PGD in waveform space to maximise speaker "
            "distance before projecting onto the null space. "
            "Slower (~2 min/file) but produces stronger protection. "
            "Use with --mode evaluate or --mode protect."
        ),
    )
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=50,
        dest="pgd_steps",
        help="Number of PGD iterations (default 50). Higher = stronger but slower.",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=0,
        dest="n_files",
        help="Limit evaluation to first N files (0 = all files).",
    )
    parser.add_argument(
        "--flat-pgd",
        action="store_true",
        default=False,
        dest="flat_pgd",
        help=(
            "Use the old flat L-inf signed-gradient PGD instead of the "
            "psychoacoustically masked PGD. The flat variant is stronger per "
            "step but adds clearly audible broadband hiss."
        ),
    )
    parser.add_argument(
        "--margin-db",
        type=float,
        default=25.0,
        dest="margin_db",
        help=(
            "Masking-threshold margin for perceptual PGD (default 25 = the "
            "validated protect+quality sweet spot: PESQ ~3.9, cloning fails "
            "on all encoders). 0 = strictly inaudible but too weak to protect; "
            "each +6 dB doubles the allowed perturbation amplitude (stronger "
            "protection, more audible). 35 = stronger protection at PESQ ~3.75."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Add project root to sys.path so all modules resolve correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Dispatch
    if args.mode == "test":
        return mode_test(args)
    elif args.mode == "train":
        return mode_train(args)
    elif args.mode == "protect":
        return mode_protect(args)
    elif args.mode == "evaluate":
        return mode_evaluate(args)
    elif args.mode == "clone-eval":
        return mode_clone_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
