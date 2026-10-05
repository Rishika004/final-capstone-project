# On-manifold vs additive protection (attacker: purifier v3 + DDIM, t=50)

## On-manifold transformations (20 held-out speakers)

| transform | magnitude | R_t (pass-through) | survival PSR | protection % | PESQ | STOI |
|---|---|---|---|---|---|---|
| eq | 4.0 | 0.000 | 0.982 | 0 | 4.54 | 0.999 |
| eq | 8.0 | 0.000 | 0.983 | 0 | 4.27 | 0.994 |
| eq | 12.0 | 0.000 | 0.984 | 0 | 3.77 | 0.982 |
| eq | 16.0 | 0.000 | 0.986 | 5 | 2.94 | 0.956 |
| reverb | 50.0 | 0.003 | 0.977 | 0 | 4.53 | 0.994 |
| reverb | 150.0 | 0.006 | 0.964 | 0 | 3.96 | 0.988 |
| reverb | 300.0 | 0.004 | 0.967 | 0 | 3.23 | 0.984 |
| reverb | 600.0 | 0.002 | 0.972 | 0 | 2.54 | 0.979 |
| tilt | -18.0 | 0.000 | 0.919 | 0 | 4.58 | 1.000 |
| tilt | -12.0 | 0.000 | 0.940 | 0 | 4.62 | 1.000 |
| tilt | -6.0 | 0.000 | 0.956 | 0 | 4.64 | 1.000 |
| tilt | 6.0 | 0.000 | 0.972 | 0 | 4.63 | 1.000 |
| tilt | 12.0 | 0.000 | 0.977 | 0 | 4.53 | 0.999 |
| tilt | 18.0 | 0.000 | 0.981 | 0 | 4.31 | 0.997 |
| warp | 0.88 | 0.002 | 0.980 | 72 | 1.18 | 0.794 |
| warp | 0.91 | 0.002 | 0.979 | 48 | 1.32 | 0.855 |
| warp | 0.94 | 0.003 | 0.976 | 32 | 1.80 | 0.911 |
| warp | 1.06 | 0.004 | 0.978 | 12 | 1.86 | 0.917 |
| warp | 1.09 | 0.002 | 0.983 | 28 | 1.36 | 0.870 |
| warp | 1.12 | 0.002 | 0.985 | 52 | 1.21 | 0.821 |

## Additive crafting, same attacker

| method | survival PSR | protection % | PESQ |
|---|---|---|---|
| purifier-aware (EOT) | 0.969 | 95 | 2.23 |
| EOT + zero-noise reg. | 0.975 | 92 | 2.25 |
| plain + noise-matched reg. | 0.983 | 65 | 2.23 |
| plain + zero-noise reg. | 0.978 | 78 | 2.22 |
| plain | 0.978 | 78 | 2.22 |

**Pearson r (survival vs protection), all methods: +0.25**
  within the on-manifold family alone: +0.31

Best on-manifold settings by protection:

- warp 0.88: protection 72 %, survival 0.980, PESQ 1.18
- warp 1.12: protection 52 %, survival 0.985, PESQ 1.21
- warp 0.91: protection 48 %, survival 0.979, PESQ 1.32
