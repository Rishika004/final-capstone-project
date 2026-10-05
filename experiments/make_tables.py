"""
experiments/make_tables.py
==========================
Aggregate exp_purifier_aware.py CSVs into the paper tables.

  python experiments/make_tables.py                       # trial CSV (purifier_aware.csv)
  python experiments/make_tables.py --csv experiments/results/main_table.csv

Outputs (next to the CSV): <name>_summary.csv and <name>_summary.md with, per
variant and evaluation strength t: protection rate per encoder (defence_valid:
clone fails after purification AND unprotected control keeps identity), the
all-encoders rate, and mean PESQ / STOI / SNR over files.
"""
import os, sys, argparse
import pandas as pd
from utils.csvio import read_csv as read_csv_safe

ENC_ORDER = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
ENC_SHORT = {"WavLM-SV": "WavLM-SV", "ECAPA-TDNN": "ECAPA", "Resemblyzer d-vector": "d-vec"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "results", "purifier_aware.csv"))
    ap.add_argument("--variants", default="", help="comma list to include (default all with eval_t)")
    args = ap.parse_args()
    df = read_csv_safe(args.csv)
    if "eval_t" not in df.columns:
        df["eval_t"] = 50
    if args.variants:
        df = df[df.variant.isin(args.variants.split(","))]
    rows = []
    for (v, t), g in df.groupby(["variant", "eval_t"], sort=False):
        q = g.drop_duplicates("file")
        r = dict(variant=v, eval_t=int(t), n_files=len(q),
                 pesq=round(q.pesq.mean(), 2), stoi=round(q.stoi.mean(), 3),
                 snr_db=round(q.snr_db.mean(), 1))
        per_file_all = g.groupby("file").defence_valid.all()
        for e in ENC_ORDER:
            ge = g[g.encoder == e]
            r[f"prot_{ENC_SHORT[e]}"] = round(100 * ge.defence_valid.mean(), 0) if len(ge) else None
            r[f"cos_after_{ENC_SHORT[e]}"] = round(ge.cos_after_purify.mean(), 3) if len(ge) else None
            r[f"cos_ctrl_{ENC_SHORT[e]}"] = round(ge.cos_ctrl_unprot.mean(), 3) if len(ge) else None
        r["prot_all"] = round(100 * per_file_all.mean(), 0)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["variant", "eval_t"])
    base = os.path.splitext(args.csv)[0]
    out.to_csv(base + "_summary.csv", index=False)
    cols = ["variant", "eval_t", "n_files", "pesq", "stoi", "snr_db",
            "prot_WavLM-SV", "prot_ECAPA", "prot_d-vec", "prot_all"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    with open(base + "_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved {base}_summary.csv / .md")


if __name__ == "__main__":
    main()
