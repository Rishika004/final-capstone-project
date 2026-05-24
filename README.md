# Null-Space Voice Protection

A proof-of-concept implementation of **null-space perturbation** for voice privacy protection.
The system adds imperceptible noise to voice recordings that:

1. **Survives DiffPure-style audio purification** — the purifier's Jacobian maps the perturbation to approximately zero.
2. **Causes AI voice cloning to fail** — speaker embeddings are pushed far from the original.
3. **Sounds normal to humans** — the perturbation respects an L∞ budget (ε ≈ 0.01).

---

## The Core Idea

DiffPure (and similar denoising purifiers) can be locally approximated as a linear operator
with Jacobian **J** at input **x**:

```
purifier(x + δ) ≈ purifier(x) + J · δ
```

If we restrict **δ** to the **null space of J**, then `J · δ ≈ 0`, meaning:

```
purifier(x + δ) ≈ purifier(x)
```

The purifier **cannot remove δ** because it literally cannot "see" it — the perturbation
is invisible to the linear approximation of the purifier.

### Algorithm

```
1. Estimate J via K randomized JVPs (Johnson-Lindenstrauss sketch)
2. Decompose J = U Σ Vᵀ  via SVD
3. Null space basis = rows of V where σᵢ < threshold
4. Generate random δ in mel space (|δ| ≤ ε)
5. Project δ onto null(J): δ_proj = Vₙᵤₗₗᵀ (Vₙᵤₗₗ δ)
6. Clamp to ε budget
7. Return x + δ_proj
```

---

## Project Structure

```
nullspace_voice_protection/
├── data/
│   └── samples/            ← put your .wav files here
├── models/
│   ├── purifier.py         ← UNet surrogate purifier (differentiable)
│   ├── cloning_proxy.py    ← surrogate voice cloning model
│   └── speaker_encoder.py  ← WavLM-Base-Plus speaker embeddings
├── core/
│   ├── jacobian.py         ← randomized Jacobian estimation (K JVPs)
│   ├── nullspace.py        ← SVD null space extraction + projection
│   ├── perturbation.py     ← single-shot null-space protection pipeline
│   └── optimizer.py        ← iterative PGD in the null space
├── evaluation/
│   ├── metrics.py          ← PRR, PESQ, cosine similarity, SNR, mel dist
│   └── evaluate.py         ← batch evaluation pipeline → results.csv
├── tests/
│   └── test_nullspace.py   ← unit tests (9 test classes)
├── notebooks/
│   └── demo.ipynb          ← end-to-end interactive demo
├── main.py                 ← CLI entry point
└── requirements.txt
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, torchaudio 2.x

---

## Quick Start

### Test the math first
```bash
python main.py --mode test
```
This runs 4 math sanity checks + the full pytest suite. All should pass.

### Protect a single file
```bash
# Put a .wav file in data/samples/ or point --input to any .wav
python main.py --mode protect \
    --input data/samples/test.wav \
    --output protected.wav \
    --epsilon 0.01
```

### Full evaluation on a directory
```bash
python main.py --mode evaluate \
    --input data/samples/ \
    --output results/ \
    --epsilon 0.01
```
Results are written to `results/results.csv`.

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | *required* | `protect`, `evaluate`, or `test` |
| `--input` / `-i` | `data/samples/` | Input .wav file or directory |
| `--output` / `-o` | `results/` | Output file or directory |
| `--epsilon` | `0.01` | L∞ perturbation budget |
| `--K` | `50` | Jacobian probe count (higher = better approximation) |
| `--svd-threshold` | `1e-3` | Singular value cutoff for null-space extraction |
| `--n-steps` | `0` | Optimiser steps (0 = single-shot, faster) |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

---

## Evaluation Metrics

| Metric | Symbol | Interpretation |
|--------|--------|----------------|
| **Purification Resistance Rate** | PRR | `||f(x+δ) - (x+δ)|| / ||f(x) - x||` — values > 1.0 mean purification has LESS effect on protected audio |
| **Speaker Cosine Similarity** | cos-sim | Cosine similarity between speaker embeddings — lower is better for defence |
| **Mel Distance** | mel-dist | L2 distance in log-mel space — lower = more imperceptible |
| **PESQ** | PESQ | Perceptual speech quality (target: > 3.0) |
| **SNR** | dB | Signal-to-noise ratio — higher = quieter perturbation |

---

## Modules in Detail

### `core/nullspace.py`
- `compute_null_space(J, threshold)` — full SVD, returns orthonormal null basis
- `project_onto_null_space(delta, null_basis)` — orthogonal projection formula
- `verify_null_space(J, null_basis)` — sanity check: prints `max|J·null_basis.T|`

### `core/jacobian.py`
- `estimate_jacobian_randomized(model, x, K=50)` — rank-K approximation via JVPs
- `compute_jacobian(model, x, method='randomized')` — unified entry point

### `core/perturbation.py`
- `apply_null_space_protection(x_mel, purifier, epsilon)` — the main pipeline

### `core/optimizer.py`
- `optimize_perturbation(...)` — iterative PGD that jointly maximises speaker distance
  and minimises perceptual distortion, re-projecting onto null(J) after each step

### `models/purifier.py`
- `MelUNetPurifier` — lightweight UNet operating on log-mel spectrograms
- `load_pretrained()` — returns a randomly initialised model (no checkpoint needed)

### `models/speaker_encoder.py`
- `WavLMSpeakerEncoder` — WavLM-Base-Plus + linear projection to 256-d
- Falls back to a dummy encoder if HuggingFace is unavailable

---

## Running Tests

```bash
pytest tests/test_nullspace.py -v
```

Test classes:
1. `TestNullSpaceDefinition` — `J @ null_basis.T ≈ 0`
2. `TestProjectionIdempotency` — `P(P(x)) == P(x)`
3. `TestProjectionOrthogonality` — projected delta lies in null(J)
4. `TestEpsilonConstraint` — `||δ||_∞ ≤ ε` after projection + clamp
5. `TestPRRNullVsRandom` — null-space PRR ≥ random PRR
6. `TestJacobianShape` — output shape is `[output_dim, input_dim]`
7. `TestEmptyNullSpaceHandled` — graceful fallback when J is full-rank
8. `TestVerifyNullSpace` — residual < tolerance
9. `TestPerturbationPipeline` — output shape and ε constraint end-to-end

---

## Sample Data

Place any 16 kHz mono `.wav` files in `data/samples/`.

For a free dataset, download a few utterances from
[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) or any freely-licensed speech corpus.

The demo notebook (`notebooks/demo.ipynb`) will automatically generate a synthetic
test signal if no `.wav` files are found.

---

## Limitations and Future Work

- **Surrogate purifier**: the UNet is randomly initialised. For a tighter approximation
  of DiffPure, train it on noisy/clean mel pairs from the LibriSpeech dataset.
- **Waveform inversion**: the current pipeline approximates the waveform-domain
  perturbation. A neural vocoder (HiFi-GAN, BigVGAN) would give a more accurate result.
- **Adaptive attacks**: this is a white-box defence against the *surrogate* purifier.
  An adaptive attacker who knows the exact null space can potentially break it —
  see the iterative optimizer in `core/optimizer.py` for a stronger variant.
- **PESQ evaluation**: install `pip install pesq` for real PESQ scores; otherwise
  the metric returns `NaN`.

---

## Citation

If you use this code in academic work, please cite:

```
@misc{nullspace_voice_protection_2026,
  title  = {Null-Space Voice Protection: Perturbations That Survive DiffPure},
  author = {[Your Name]},
  year   = {2026},
  note   = {Proof-of-concept implementation}
}
```
