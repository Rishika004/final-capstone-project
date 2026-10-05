"""
training/train_purifier_v2.py
=============================
Multi-speaker, DDPM-consistent retraining of the mel denoiser used as the
DiffPure backbone (models/real_diffpure.py).

Why v2
------
The v1 checkpoint (checkpoints/purifier_trained.pt) was trained on LJSpeech
only (one speaker) with fixed additive noise. Diagnostics (2026-09-10):
  * single-pass reconstruction error on unseen LibriSpeech speakers is 2-3x
    the in-domain error, and
  * the 50-step DDPM loop compounds that bias into ~2 nats mean log-mel error,
    which destroys speaker identity even on UNPROTECTED audio (channel EER
    14-23 % vs < 4 % clean).
An attacker would never use a purifier that erases the identity they want to
steal, so every end-to-end number built on v1 is meaningless.

v2 changes
----------
  * data: LibriSpeech dev-clean speakers [0:20) (sorted ids) + LJSpeech.
          Speakers [20:40) are NEVER seen in training -> used for evaluation.
  * noise: exactly the DDPM forward process the purifier will run at test
          time,  x_t = sqrt(abar_t) x + sqrt(1-abar_t) eps,  t ~ U{0..T_MAX},
          same linear beta schedule as MelDiffPure. Target = x_0.
  * on-the-fly random crops + noise every step (no fixed dataset).
  * same MelUNetPurifier architecture -> drop-in checkpoint.

Run:
  python training/train_purifier_v2.py --steps 12000 --out checkpoints/purifier_libri.pt
"""
import sys, io, os, argparse, time, glob
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import torch.nn as nn

from models.purifier import MelUNetPurifier, MelUNet2D
from models.real_diffpure import MelDiffPure
from evaluation.evaluate import load_wav, wav_to_logmel
from experiments.librispeech import (list_librispeech, per_speaker, load_flac,
                                     TRAIN_SUBSET)

N_STEPS_DDPM = 1000


def load_mels(device, n_train_spk, n_lj, max_sec, max_utt_per_spk=20):
    # TRAIN ON A DIFFERENT SPLIT FROM THE ONE WE EVALUATE ON (changed 2026-09-17).
    # Previously this was "dev-clean" -- the SAME 40-speaker split used for
    # evaluation, with the purifier taking speakers [0:20) and every reported
    # number coming from [20:40). That is same-split train/eval, and De-AntiFake
    # explicitly guards against it ("These speakers do not overlap with those in
    # the training set of the purification models") by training on train-clean-100.
    # Training here frees ALL 40 test-clean speakers for evaluation and lets us
    # state the same disjointness guarantee.
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             TRAIN_SUBSET, min_sec=2.0)
    # Cap utterances per speaker. On dev-clean (40 speakers) taking everything was
    # fine; train-clean-100 has 251 speakers and ~28k utterances, which will not fit
    # in memory as mels. Capping trades depth per speaker for BREADTH across
    # speakers, which is what a purifier should generalise over.
    by = per_speaker(items, n_utt=max_utt_per_spk)
    spks = sorted(by)
    # LEAK FIX 2026-09-16: val speakers used to be spks[n_train_spk:n_train_spk+4],
    # i.e. the FIRST FOUR EVALUATION SPEAKERS (3000/3081/3170/3536). Checkpoint
    # selection therefore peeked at the protection-evaluation population. Hold
    # out the validation speakers from the TRAINING pool instead, leaving
    # spks[n_train_spk:] entirely untouched.
    train_spk = spks[:n_train_spk - 4]
    val_spk = spks[n_train_spk - 4:n_train_spk]
    print(f"[v2] train speakers {train_spk[0]}..{train_spk[-1]} ({len(train_spk)}) | "
          f"val speakers {val_spk} | split={TRAIN_SUBSET} | "
          f"evaluation happens on a DIFFERENT split entirely")
    train, val = [], []
    for s in train_spk:
        for it in by[s]:
            train.append(wav_to_logmel(load_flac(it["path"], max_sec).to(device), device).squeeze(0))
    for s in val_spk:
        for it in by[s][:5]:
            val.append(wav_to_logmel(load_flac(it["path"], max_sec).to(device), device).squeeze(0))
    lj = sorted(glob.glob(os.path.join(PROJ, "data", "LJSpeech-1.1", "wavs", "*.wav")))[:n_lj]
    for f in lj:
        train.append(wav_to_logmel(load_wav(f, max_seconds=max_sec).to(device), device).squeeze(0))
    print(f"[v2] {len(train)} train mels ({len(lj)} LJSpeech) | {len(val)} val mels")
    return train, val


def random_batch(mels, bs, crop, gen):
    xs = []
    for _ in range(bs):
        m = mels[int(torch.randint(len(mels), (1,), generator=gen))]
        T = m.shape[-1]
        if T < crop:
            m = nn.functional.pad(m, (0, crop - T), value=float(m.min()))
            T = crop
        o = int(torch.randint(T - crop + 1, (1,), generator=gen))
        xs.append(m[:, o:o + crop])
    return torch.stack(xs)


def ddpm_noise(x0, abar, t_max, gen_dev):
    B = x0.shape[0]
    t = torch.randint(0, t_max + 1, (B,), generator=gen_dev, device=x0.device)
    ab = torch.where(t == 0, torch.ones_like(t, dtype=abar.dtype), abar[(t - 1).clamp(min=0)])
    ab = ab.view(B, 1, 1)
    eps = torch.randn(x0.shape, generator=gen_dev, device=x0.device)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


@torch.no_grad()
def validate(model, val, abar, device, gen_dev, t_eval=50):
    """Returns (clean identity L1, DDPM-loop L1 on clean, denoising gain).
    denoising gain = 1 - L1(model(x_t), x) / L1(x_t, x) at t=t_eval single pass:
    0 = the model does not denoise at all (identity), 1 = perfect."""
    model.eval()
    dp = MelDiffPure(model, n_steps=N_STEPS_DDPM, t_diffuse=t_eval).to(device)
    e1, e2, gain = [], [], []
    g = torch.Generator(device=device).manual_seed(123)
    ab = abar[t_eval - 1]
    for m in val:
        x = m.unsqueeze(0)
        e1.append((model(x) - x).abs().mean().item())
        e2.append((dp(x, seed=1) - x).abs().mean().item())
        xt = ab.sqrt() * x + (1 - ab).sqrt() * torch.randn(x.shape, generator=g, device=device)
        gain.append(1.0 - (model(xt) - x).abs().mean().item() / (xt - x).abs().mean().item())
    model.train()
    return sum(e1) / len(e1), sum(e2) / len(e2), sum(gain) / len(gain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--t-max", type=int, default=100, dest="t_max")
    ap.add_argument("--n-train-spk", type=int, default=64, dest="n_train_spk",
                    help="train-clean-100 has 251; 64 gives far more speaker "
                         "diversity than the old 20 without blowing memory")
    ap.add_argument("--max-utt-per-spk", type=int, default=20, dest="max_utt_per_spk")
    ap.add_argument("--n-lj", type=int, default=600, dest="n_lj")
    ap.add_argument("--max-sec", type=float, default=6.0, dest="max_sec")
    ap.add_argument("--init", default="", help="optional v1 checkpoint to warm-start from")
    ap.add_argument("--out", default=os.path.join(PROJ, "checkpoints", "purifier_libri.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual", action="store_true", help="predict x0 = x + UNet(x)")
    ap.add_argument("--arch", default="unet1d", choices=["unet1d", "unet2d"])
    ap.add_argument("--base", type=int, default=64, help="base channels")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    gen_dev = torch.Generator(device=device).manual_seed(args.seed)

    train, val = load_mels(device, args.n_train_spk, args.n_lj, args.max_sec,
                           args.max_utt_per_spk)
    betas = torch.linspace(1e-4, 0.02, N_STEPS_DDPM, device=device)
    abar = torch.cumprod(1.0 - betas, dim=0)

    model = (MelUNet2D(n_mels=80, base_channels=args.base, residual=args.residual)
             if args.arch == "unet2d" else
             MelUNetPurifier(n_mels=80, base_channels=args.base, residual=args.residual)).to(device)
    print(f"[v2] arch={args.arch} base={args.base} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    if args.init and os.path.isfile(args.init):
        ck = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ck.get("model_state_dict", ck))
        print(f"[v2] warm start from {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    e1, e2, gn = validate(model, val, abar, device, gen_dev)
    print(f"[v2] step 0 | val single-pass L1={e1:.3f} | val DDPM(t=50) L1={e2:.3f} | denoise gain={gn:.2f}")

    best, t0 = float("inf"), time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x0 = random_batch(train, args.batch, args.crop, gen)
        xt = ddpm_noise(x0, abar, args.t_max, gen_dev)
        pred = model(xt)
        loss = nn.functional.l1_loss(pred, x0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            e1, e2, gn = validate(model, val, abar, device, gen_dev)
            print(f"[v2] step {step:5d} | train L1={loss.item():.3f} | val single={e1:.3f} "
                  f"| val DDPM50={e2:.3f} | denoise gain={gn:.2f} | {time.time() - t0:.0f}s", flush=True)
            score = e2 - gn          # want low loop error AND high denoising gain
            if score < best:
                best = score
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "step": step,
                            "val_single_l1": e1, "val_ddpm50_l1": e2, "val_denoise_gain": gn,
                            "train_speakers": args.n_train_spk, "t_max": args.t_max,
                            "n_mels": 80, "base_channels": args.base, "residual": args.residual,
                            "arch": args.arch}, args.out)
    print(f"[v2] done. best (loopL1 - gain)={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
