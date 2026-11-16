# Pre-registration — N7, the held-out contrast at matched budget

**Written 2026-11-12, before the run.** The experiment RESULTS §61d names as
the highest-value remaining one, and the one a reviewer asks for first.

---

## 1. The gap, stated exactly

Four conditions have now been measured and they do not agree:

| # | condition | budget | verdict on purifier-in-the-loop |
|---|---|---|---|
| §59b | matched ensemble, `m30_f15`, DiffVC | 0.86 dB apart | 0 of 9; 4 cells saturated |
| §61b | matched ensemble, **seg**, DiffVC | **matched** (0.093 dB, p = 0.41) | 0 of 9 favourable; **2 of 9 against**, on the d-vector |
| §59c | **held out**, `m30_f15`, DiffVC | **2.89 dB apart** | 7 of 9 favourable — **confounded** |
| §60d | matched ensemble, **seg**, F5-TTS | **matched** | 8 of 9 favourable, 5 significant |

The claim the paper wants to make — purifier-in-the-loop crafting buys
**coverage**, i.e. it helps against what was not targeted — lives in the
**held-out** row. That row is the only one whose budget is **not** matched, and
§59f measured the mismatch at 2.89 dB.

`plain_m30_seg_nodvec` does not exist on disk. So the held-out claim has never
been tested at matched budget, in any condition, against any cloner.

## 2. Why the budgets should match once this is crafted

§60d established that **segment crafting is what makes the plain arm spend its
budget**: plain goes 19.02 dB (`m30_f15`) → 16.21 dB (`seg`), landing on top of
the purifier-aware arm's 16.11 dB. A per-frame diagnostic agrees — the seg
arms sit at the `--snr-floor 15` gate in ~84 % of non-silent frames.

The prediction is therefore that `plain_m30_seg_nodvec` lands near 16.1–16.2 dB
and matches `fullnsT_m30_seg_nodvec` without any tuning. **This is checked, not
assumed** (§4), and if it fails the run is reported as a failed match.

## 3. What is run

**Craft.** `experiments/exp_purifier_aware.py`, identical flags to
`run_heldout_seg.sh`'s `full_m30_seg_nodvec` except `--variant plain`:

```
--dataset librispeech --subset test-clean --cross-session --spk-offset 0
--utt-index 0 --n-files 29 --steps 200 --purifier-ckpt checkpoints/purifier_v7.pt
--sampler ddim --thresholds <v7 ddim t50 xsess tc> --eval-t 50,200
--thr-aware --weights 1,1,1 --seg-partials --craft-encoders no-dvec
--variant plain --margin-db 30 --snr-floor 15 --tag plain_m30_seg_nodvec
```

**Evaluate.** DiffVC cloning, all three arms in **one** run so the comparison
is within-run (§31b: 32 % of unprotected cells flip between identical runs, so
a cross-run binary comparison is not safe):
`plain_m30_seg_nodvec`, `full_m30_seg_nodvec`, `fullnsT_m30_seg_nodvec`,
channels `none` / `ours` / `diffwave` (`--dw-t 3`), 29 cross-session speakers,
cross-session EER thresholds. Output `results/cloning_heldout_seg_n7.csv`.

**Optionally, and only if the DiffVC rows are informative:** the same three
arms through F5-TTS, reusing the N2 harness unchanged. Decided after the
DiffVC rows are read, and reported as exploratory if run.

### 3a. Amendment, before the run: `--clone-seed 2026`

§62b (written after §3 above, on the same day) measured DiffVC's run-to-run
clone noise on **protected** arms: the same contrast, same script, different
arm list, moved 0.0137 in the margin and two speakers in the binary rate,
because the clones are unseeded and the arm list shifts RNG consumption. The
rule §62b sets is that every new cloning run passes `--clone-seed`.

This run therefore passes **`--clone-seed 2026`**. The seed is fixed here,
before the data, and mixed per (speaker, arm, channel) by the script's own CRC
derivation, so it cannot be chosen to favour an outcome. It makes the run
reproducible; it does not change what is being compared, which is already
protected by all three arms being in one run.

## 4. The matching check — first, and it gates everything

Realised waveform SNR and wideband PESQ against the original 4-s crop,
`plain_m30_seg_nodvec` vs `fullnsT_m30_seg_nodvec`, paired over 29 speakers
(`experiments/analyse_n1_budget.py`, extended to these arms).

> **GATE:** the paired SNR difference must be **within 0.5 dB** and its paired
> t-test **not significant at 0.05**. If either fails, the arms are not
> budget-matched, the contrast is reported as **still confounded**, and §59c
> keeps its §59f wording. No verdict is read from a failed match.

## 5. Primary endpoint

`fullnsT_m30_seg_nodvec` **minus** `plain_m30_seg_nodvec`, paired per speaker,
**ECAPA-TDNN**, channel **`ours`**, n = 29. Paired t, BCa interval.

**Chosen with §61b's error in mind.** The N5 primary cell was saturated and
could not resolve anything. Here both arms are *expected* to leave headroom,
because the held-out `m30_f15` cells ran at 10–28 % clone success rather than
0 %. **The saturation check is therefore part of the endpoint:** if both arms
sit below 10 % clone success in the primary cell, the cell is declared
**not identifiable** and the primary moves — pre-specified here, before the
data — to **WavLM-SV at `ours`**, and if that is also floored, to the
**continuous margin on the d-vector at `ours`**, which has never saturated in
any held-out cell. Secondary: the other eight cells, Holm within the nine.

## 6. Branches, fixed before the data

* **7-A — the coverage claim survives at matched budget.** Primary cell
  significantly negative.
  → This is the paper's lead result, now budget-matched and un-confounded, and
  §59c's wording can drop the 2.89 dB caveat for the seg arms.
* **7-B — direction holds, significance does not.**
  → Report the attenuated effect with its interval. The claim becomes "the
  F5-TTS evidence (§60d) carries the coverage result; the DiffVC held-out
  evidence is directionally consistent and underpowered".
* **7-C — the effect disappears or reverses at matched budget.**
  → **§59c was a budget effect.** The coverage claim then rests on §60d alone,
  i.e. on varying the cloner rather than the ensemble, and the paper must say
  that plainly. §59 is amended.
* **7-D — the match fails** (§4).
  → No verdict; report as still confounded.

**Pre-committed prediction.** I expect **7-B**, at about 45 %, with 7-A at 30 %
and 7-C at 25 %. Reasoning: §59c's held-out effect was large (ECAPA −0.0576)
and the two indirect tests in §59f pointed away from magnitude, which argues
for the effect being real; but §61b showed that when the budget is genuinely
matched the DiffVC advantage vanished in the matched condition, and the
held-out condition differs only in which encoders were targeted. My last two
predictions in this project (§7.5, §60c) were both wrong, so this one deserves
little weight.

## 7. Cost

Plain seg crafting at 29 speakers (plain is the cheap variant; the EOT arms ran
~6 min/speaker, plain should be well under that) plus one 4-variant cloning run
(~37 min measured). Estimate **2–4 GPU-h**, one job at a time.

## 8. Files

| what | path |
|---|---|
| craft + evaluate | `experiments/run_n7_heldout_matched.sh` |
| budget check | `experiments/analyse_n1_budget.py --arms <the seg nodvec pair>` |
| analysis | `experiments/analyse_n5.py --in results/cloning_heldout_seg_n7.csv --ship fullnsT_m30_seg_nodvec --plain plain_m30_seg_nodvec` |
