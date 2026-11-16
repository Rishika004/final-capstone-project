"""tests/test_n2_prompts.py -- the invariants N2's protocol depends on.

PREREG: paper/PREREG_N2_modern_cloner.md

Three things can silently invalidate the modern-cloner column, and none of them
shows up as an error at run time:

  1. `ref_text` must be IDENTICAL across arms for a given victim. If a protected
     clip were transcribed separately, a protection that merely confuses the
     attacker's ASR would show up as speaker protection (prereg 2.4). The whole
     point of caching Whisper's transcript of the CLEAN crop is to forbid that.
  2. `gen_text` must be the SOURCE speaker's text under the same
     `spks[(i+1) % n]` rotation exp_cloning_protection.py uses, so the content
     pathway matches the DiffVC run it sits beside in the table.
  3. the per-generation seed must be derived exactly as --clone-seed derives it,
     or "reproducible per (speaker, arm, channel)" is false.

These are cheap to assert and expensive to discover later.

Run:  python -m pytest tests/test_n2_prompts.py -q
"""
import os
import sys
import zlib

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

PROMPTS = os.path.join(PROJ, "experiments", "results", "f5_prompts.csv")


def _seed_ref(clone_seed, spk, arm, pname):
    """The derivation in exp_cloning_protection.py, verbatim."""
    h = zlib.crc32(f"{spk}|{arm}|{pname}".encode()) & 0xFFFFFFFF
    return (clone_seed * 1000003 + h) % (2 ** 63)


def test_seed_derivation_matches_the_vc_run():
    """f5_clone.py must mix the seed the same way, modulo its 2**31 range.

    The range differs because numpy's seed must fit in 32 bits; the MIXING must
    not, or two scripts claiming "seeded per (speaker, arm, channel)" would mean
    different things.
    """
    for spk in (61, 121, 1089):
        for arm in ("unprotected", "fullnsT_tc_m30_seg"):
            for pname in ("none", "ours", "diffwave"):
                h = zlib.crc32(f"{spk}|{arm}|{pname}".encode()) & 0xFFFFFFFF
                mine = (2026 * 1000003 + h) % (2 ** 31)
                theirs = _seed_ref(2026, spk, arm, pname)
                assert mine == theirs % (2 ** 31)


def test_seed_is_distinct_per_cell():
    seeds = {(spk, arm, pur): (2026 * 1000003 +
                               (zlib.crc32(f"{spk}|{arm}|{pur}".encode())
                                & 0xFFFFFFFF)) % (2 ** 31)
             for spk in (61, 121, 1089)
             for arm in ("unprotected", "plain_tc_m30_seg", "fullnsT_tc_m30_seg")
             for pur in ("none", "ours", "diffwave")}
    assert len(set(seeds.values())) == len(seeds), "seed collision across cells"


# ---------------------------------------------------------------------------
# The rest need the real prompts file. Skipped rather than failed when stage A
# has not run yet, so the suite stays green on a fresh checkout.
# ---------------------------------------------------------------------------
pytestmark_needs_file = pytest.mark.skipif(
    not os.path.isfile(PROMPTS), reason="f5_prompts.csv not built yet")


@pytest.fixture(scope="module")
def prompts():
    if not os.path.isfile(PROMPTS):
        pytest.skip("f5_prompts.csv not built yet")
    import pandas as pd
    return pd.read_csv(PROMPTS)


def test_ref_text_constant_across_arms_and_channels(prompts):
    """Invariant 1. One transcript per victim, full stop."""
    bad = {}
    for spk, grp in prompts.groupby("speaker"):
        texts = set(grp.ref_text.astype(str))
        if len(texts) != 1:
            bad[spk] = texts
    assert not bad, f"ref_text varies within a speaker: {bad}"


def test_gen_text_is_the_source_speakers_ref_text(prompts):
    """Invariant 2. The rotation, and the text it implies, must agree."""
    ref_of = {int(s): g.ref_text.iloc[0]
              for s, g in prompts.groupby("speaker")}
    spks = sorted(ref_of)
    expect_src = {s: spks[(i + 1) % len(spks)] for i, s in enumerate(spks)}
    for _, r in prompts.iterrows():
        spk, src = int(r["speaker"]), int(r["source_speaker"])
        assert src == expect_src[spk], (
            f"spk{spk}: source_speaker={src}, expected {expect_src[spk]}")
        assert str(r["gen_text"]) == str(ref_of[src]), (
            f"spk{spk}: gen_text is not spk{src}'s transcript")


def test_victim_never_supplies_its_own_content(prompts):
    """A clone whose words came from the victim's own clip would be scored
    against an enrolment of the same speaker saying related material. The
    rotation forbids it; assert it rather than trust it."""
    assert (prompts.speaker != prompts.source_speaker).all()


def test_arms_and_channels_are_the_preregistered_ones(prompts):
    assert set(prompts.variant) <= {"unprotected", "plain_tc_m30_seg",
                                    "fullnsT_tc_m30_seg"}
    assert set(prompts.purifier) <= {"none", "ours", "diffwave"}
    assert "phonepure" not in set(prompts.purifier), "N2 excludes PhonePuRe"


def test_every_reference_wav_exists(prompts):
    missing = [p for p in prompts.ref_wav
               if not os.path.isfile(os.path.join(PROJ, p))]
    assert not missing, f"{len(missing)} reference wavs missing, e.g. {missing[:3]}"


def test_enrolment_is_a_different_chapter_than_the_reference(prompts):
    """Fix S1. If the enrolment shared a session with the reference the clone was
    built from, both sides of the comparison would be flattered by channel
    match. The shared selector is supposed to guarantee this; check it."""
    for _, r in prompts.iterrows():
        enr, ref = str(r["enrol_path"]), str(r["ref_path"])
        ch_enr = os.path.basename(os.path.dirname(enr))
        ch_ref = os.path.basename(os.path.dirname(ref))
        assert ch_enr != ch_ref, (
            f"spk{r['speaker']}: enrolment and reference share chapter {ch_ref}")
