"""
experiments/validate_stride.py
==============================
Is strided-DDIM crafting safe to use in a reported run?

WHY THIS EXISTS
---------------
`--craft-stride k` subsamples the DDIM trajectory during crafting, cutting the
dominant cost ~k-fold. DDIM is deterministic and explicitly supports trajectory
subsampling, so this should be sound. "Should be" is not evidence.

TWO THINGS CAN GO WRONG, AND THEY ARE DIFFERENT
-----------------------------------------------
  (1) FORWARD FIDELITY. The strided purifier output may differ from the full one.
      If it does, the optimiser is looking at a different attacker than the one
      we evaluate against.

  (2) GRADIENT FIDELITY. The output can match while the GRADIENT does not, and
      the gradient is what actually steers crafting. A high cosine between the
      two gradients is the thing that licenses the speedup.

(2) is the one that matters and the one an eyeball test would miss.

The honest check is end-to-end protection, run below as stage 3: craft the same
speakers at stride 1 and stride k, evaluate BOTH through the full-resolution
attacker, and compare protection. If protection holds, the speedup is free.

Run:  python experiments/validate_stride.py --strides 1,5,10 --n-files 3
"""
import sys, io, os, argparse, time

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import torch
import torch.nn.functional as F


def cosine(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    return float(F.cosine_similarity(a, b).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strides", default="1,5,10")
    ap.add_argument("--n-files", type=int, default=3, dest="n_files")
    ap.add_argument("--purifier-ckpt", default="checkpoints/purifier_v7.pt",
                    dest="ckpt")
    ap.add_argument("--t-diffuse", type=int, default=50, dest="t_diffuse")
    a = ap.parse_args()
    strides = [int(s) for s in a.strides.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.real_diffpure import load_real_diffpure
    from evaluation.evaluate import wav_to_logmel
    from experiments.exp_purifier_aware import diffpure_grad
    from experiments.librispeech import list_librispeech, reference_items, load_flac

    dp = load_real_diffpure(checkpoint_path=a.ckpt, t_diffuse=a.t_diffuse,
                            device=device, sampler="ddim")
    for p in dp.parameters():
        p.requires_grad_(False)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    spks = sorted(ref)[: a.n_files]

    print(f"purifier {os.path.basename(a.ckpt)}, t={a.t_diffuse}, ddim, "
          f"{len(spks)} speakers, device={device}\n")
    print("STAGE 1+2 — forward output and GRADIENT, strided vs full")
    print("The gradient column is the one that licenses the speedup.\n")
    print(f"  {'stride':>7}{'passes':>8}{'time/step':>11}{'speedup':>9}"
          f"{'output cos':>12}{'GRADIENT cos':>14}")

    base_grad, base_out, base_t = {}, {}, None
    for k in strides:
        outs, grads, ts = [], [], []
        for spk in spks:
            wav = load_flac(ref[spk]["path"], 4.0).to(device)
            mel = wav_to_logmel(wav, device).detach().requires_grad_(True)
            gen = torch.Generator(device=device); gen.manual_seed(1234)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            out = diffpure_grad(dp, mel, gen, stride=k)
            # a scalar that depends on the whole output, so the gradient is the
            # full Jacobian-vector product the optimiser would actually use
            loss = (out ** 2).mean()
            g, = torch.autograd.grad(loss, mel)
            torch.cuda.synchronize() if device.type == "cuda" else None
            ts.append(time.time() - t0)
            outs.append(out.detach().flatten().cpu())
            grads.append(g.detach().flatten().cpu())
        if k == strides[0]:
            base_out, base_grad, base_t = outs, grads, float(np.mean(ts))
            oc = gc = 1.0
        else:
            oc = float(np.mean([cosine(o, b) for o, b in zip(outs, base_out)]))
            gc = float(np.mean([cosine(g, b) for g, b in zip(grads, base_grad)]))
        n_passes = len(list(range(a.t_diffuse, 0, -k))) + (
            0 if (a.t_diffuse % k == 1 or k == 1) else 1)
        sp = base_t / float(np.mean(ts))
        print(f"  {k:>7}{n_passes:>8}{np.mean(ts):>10.2f}s{sp:>8.1f}x"
              f"{oc:>12.4f}{gc:>14.4f}")

    print("\nHOW TO READ IT")
    print("  gradient cos > 0.99  -> the strided gradient points essentially the")
    print("                          same way. Safe; use it.")
    print("  0.95 - 0.99          -> usable, but confirm with stage 3 below before")
    print("                          committing a reported run to it.")
    print("  < 0.95               -> do NOT use. The optimiser would be following a")
    print("                          different attacker than the one we evaluate.")
    print("\nSTAGE 3 (the honest check) — craft at each stride, evaluate BOTH")
    print("through the FULL-resolution attacker, compare protection:")
    print("  for K in 1 5; do")
    print("    python experiments/exp_purifier_aware.py --dataset librispeech \\")
    print("      --subset test-clean --cross-session --spk-offset 0 --utt-index 0 \\")
    print("      --n-files 3 --steps 200 --purifier-ckpt checkpoints/purifier_v7.pt \\")
    print("      --sampler ddim --thresholds 0.874,0.307,0.682 --eval-t 50 \\")
    print("      --thr-aware --weights 1,1,1 --craft-encoders all --seg-partials \\")
    print("      --variant full --vocoder phase --margin-db 30 --snr-floor 15 \\")
    print("      --lam-null 10 --null-t 200 --craft-stride $K \\")
    print("      --tag STRIDE$K --out experiments/results/_stride_check.csv")
    print("  done")


if __name__ == "__main__":
    main()
