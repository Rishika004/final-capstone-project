"""
make_share.py -- build a folder a teammate can be handed.

  python make_share.py [--out <dir>] [--zip]

WHAT GOES IN, AND WHAT DELIBERATELY DOES NOT
--------------------------------------------
IN   all our own code, the paper material, every result CSV the claims rest
     on, the full lab record, the figures, and purifier_v7.pt (12 MB) -- we
     trained that one, so nobody can download it.

OUT  and each for a reason, stated in the bundle's README rather than left
     as a gap:
       data/            LibriSpeech, gigabytes, publicly downloadable
       checkpoints/*    1.1 GB; only purifier_v7.pt is ours and needed
       models/ weights  speechbrain / HF caches, downloaded on first run
       external/        De-AntiFake, NISQA, Coqui TTS -- third-party code
                        under their own licences; not ours to redistribute
       *.wav            ~90 MB of generated audio, reproducible from the code
       the root PDF     AntiFake (CCS 2023), copyrighted, not ours to pass on

WHERE IT IS WRITTEN. Outside the project by default, because the project sits
in OneDrive and bulk writes there cost 2-3x on every later GPU run (that cost
a session today). The default lands on the Desktop, next to the project, so it
is easy to find and easy to share.
"""
import argparse, os, shutil, sys, fnmatch, datetime

PROJ = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop",
                           "Phantom", "phantom_paper_share")

CODE_DIRS = ["experiments", "core", "models", "utils", "evaluation",
             "training", "tests", "demo"]
CODE_PAT = ("*.py", "*.sh", "*.md")
SKIP_DIRS = {"__pycache__", ".pytest_cache", "results", "runs", "clones",
             "f5_clones", "f5_refs", "purifier_aware", ".git"}
ROOT_FILES = ["requirements.txt", "main.py", "README.md"]
FIGS = ["fig_n1_coverage.png", "fig_n1_coverage.csv",
        "fig_cloners.png", "fig_cloners.csv", "fig1_alpha.png"]
CKPT = os.path.join("checkpoints", "purifier_v7.pt")


def copy_tree(src, dst, patterns):
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if not any(fnmatch.fnmatch(f, p) for p in patterns):
                continue
            s = os.path.join(root, f)
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.abspath(PROJ) in out:
        sys.exit("refusing to write the bundle inside the project itself")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    tally = {}

    # ---- code ----------------------------------------------------------
    n = 0
    for d in CODE_DIRS:
        src = os.path.join(PROJ, d)
        if os.path.isdir(src):
            n += copy_tree(src, os.path.join(out, "code", d), CODE_PAT)
    for f in ROOT_FILES:
        s = os.path.join(PROJ, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out, "code", f)); n += 1
    tally["code files"] = n

    # ---- paper ---------------------------------------------------------
    n = copy_tree(os.path.join(PROJ, "paper"), os.path.join(out, "paper"),
                  ("*.md", "*.bib", "*.tex"))
    os.makedirs(os.path.join(out, "paper", "figures"), exist_ok=True)
    for f in FIGS:
        s = os.path.join(PROJ, "paper", f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out, "paper", "figures", f)); n += 1
    tally["paper files"] = n

    # ---- results -------------------------------------------------------
    rdst = os.path.join(out, "results")
    os.makedirs(os.path.join(rdst, "csv"), exist_ok=True)
    n = 0
    rsrc = os.path.join(PROJ, "experiments", "results")
    for f in sorted(os.listdir(rsrc)):
        p = os.path.join(rsrc, f)
        if os.path.isfile(p) and f.endswith((".csv", ".txt")):
            shutil.copy2(p, os.path.join(rdst, "csv", f)); n += 1
    for f in ("RESULTS_ICASSP.md", "THREAT_MODEL.md"):
        s = os.path.join(PROJ, "experiments", f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(rdst, f)); n += 1
    tally["result files"] = n

    # ---- the one checkpoint that is ours -------------------------------
    s = os.path.join(PROJ, CKPT)
    if os.path.isfile(s):
        os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
        shutil.copy2(s, os.path.join(out, "checkpoints", "purifier_v7.pt"))
        tally["checkpoint"] = "purifier_v7.pt (%.0f MB)" % (
            os.path.getsize(s) / 2 ** 20)

    # ---- the listening test, minus the audio ---------------------------
    lt = os.path.join(PROJ, "listening_test")
    if os.path.isdir(lt):
        d = os.path.join(out, "listening_test")
        os.makedirs(d, exist_ok=True)
        for f in ("index.html", "stimuli_key.csv", "stimuli_manifest.json",
                  "design.json"):
            s = os.path.join(lt, f)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(d, f))

    rs = os.path.join(PROJ, "paper", "README_SHARE.md")
    if os.path.isfile(rs):
        shutil.copy2(rs, os.path.join(out, "README.md"))
        os.remove(os.path.join(out, "paper", "README_SHARE.md"))

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    tally["total size"] = "%.0f MB" % (total / 2 ** 20)
    tally["built"] = datetime.date.today().isoformat()

    with open(os.path.join(out, "MANIFEST.txt"), "w", encoding="utf-8") as fh:
        for k, v in tally.items():
            fh.write(f"{k}: {v}\n")
    for k, v in tally.items():
        print(f"  {k}: {v}")
    print(f"\n-> {out}")

    if a.zip:
        z = shutil.make_archive(out, "zip", out)
        print(f"-> {z} ({os.path.getsize(z)/2**20:.0f} MB)")


if __name__ == "__main__":
    main()
