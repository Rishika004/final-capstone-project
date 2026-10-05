# Formal Threat Model

*Null-Space Voice Protection Against AI Voice Cloning*

This section formalises the parties, their knowledge and capabilities, and the
precise conditions under which the defence succeeds or fails. The structure
follows the convention used in recent voice-protection security papers
(e.g. AntiFake, CCS 2023; VocalBridge, 2026).

---

## 1. Parties

The system involves three parties.

**User (Defender).** Owns a clean utterance **x** and wishes to publish it
(social media, voicemail, podcast) without enabling unauthorized voice cloning.
Before release, the user applies the protection function **P** to obtain a
protected utterance:

  x_p = P(x)

**Attacker (Adversary).** Collects the protected utterance x_p and attempts to
synthesize speech **x̃** that reproduces the user's vocal identity. The attacker
may first apply a purification function **R** (e.g. DiffPure) to remove the
protective perturbation, then feed the result to a cloning model **G**
(TTS or voice-conversion):

  x_r = R(x_p)      (purification)
  x̃  = G(x_r)      (cloning)

**Verifier (Speaker Encoder / ASV).** Maps a waveform to a speaker embedding
**e = E(·)** and judges identity by cosine similarity against an enrollment
embedding e_enroll. A clone is *accepted* (attack succeeds) when

  cos( E(x̃), e_enroll ) > τ ,   with τ = 0.5 in our evaluation.

In our pipeline the speaker encoder is WavLM-Base-Plus (94.6 M parameters,
256-dim embedding).

---

## 2. Knowledge Assumptions

| Component | User knows | Attacker knows |
|---|---|---|
| Clean audio x | Yes | No (only x_p) |
| Protection P and its parameters ψ | Yes (white-box) | No (black-box) |
| Purifier R (DiffPure) | Yes (white-box surrogate) | Yes (white-box) |
| Cloning model G | No | Yes (white-box) |
| Speaker encoder E | Yes (white-box, target of PGD) | Yes |

The defender's advantage is **white-box access to the purifier**. This is the
realistic and defensible assumption: diffusion purifiers such as DiffPure are
public, pretrained, and plug-and-play, so a defender can analyse the exact model
an attacker would download. The defender does **not** need access to the
attacker's specific cloning model G.

The attacker's advantage is white-box access to G and E, and the ability to
apply any purifier R. The attacker does **not** have access to the clean audio x
or to the internal parameters of the protection P.

---

## 3. Attacker Capabilities

The adversary operates under realistic, bounded capabilities:

1. **Collection.** Obtain protected utterances {x_p} of the target.
2. **Purification.** Apply a pretrained purifier R (DiffPure DDPM loop,
   t_diffuse = 50 in our evaluation) to attempt removal of the perturbation.
3. **Cloning.** Run any zero-shot / few-shot TTS or VC model G on the purified
   audio.
4. **No white-box access to P.** The attacker cannot read the protection's
   parameters, gradients, or the random seed used to generate the perturbation.

---

## 4. Defender Objectives

The protection P must satisfy three properties.

**(a) Imperceptibility.** The perturbation must be inaudible. Formally, the
L∞ budget in mel space is bounded by ε (ε = 0.01 in the main configuration),
and perceptual quality must remain high: PESQ > 3.0, STOI > 0.80,
mel-space SNR > 30 dB.

**(b) Purification-resistance.** The perturbation must survive R. Formally, the
mel-space perturbation δ must lie in the **null space of the purifier's
Jacobian J = ∂R/∂x**, so that

  J · δ ≈ 0   ⇒   R(x + δ) ≈ R(x) + δ ,

i.e. the purifier cannot distinguish or remove δ. We quantify this with the
Purification Resistance Rate (PRR) and with the *range-energy fraction*
‖P_range(δ)‖² / ‖δ‖², which measures the portion of δ the purifier can erase.

**(c) Inimitability.** After purification and cloning, the synthesized speech
must be rejected by the verifier:

  cos( E(x̃), e_enroll ) < τ .

We approximate this end-to-end objective by driving the speaker cosine
similarity between the protected and original audio below τ via PGD against E.

---

## 5. Attack Success / Defence Success

Let the enrollment embedding be e_enroll = E(x).

- **Defence succeeds** iff, after the attacker applies R and G,
  cos( E(x̃), e_enroll ) < τ  AND the perturbation satisfied the
  imperceptibility budget (‖δ‖∞ ≤ ε).
- **Attack succeeds** iff cos( E(x̃), e_enroll ) ≥ τ, i.e. purification restored
  enough speaker identity for the clone to be accepted.

This yields the primary security claim: because δ lies in null(J), purification
leaves it intact (PRR ≈ 1, range-energy fraction ≈ 0), so the speaker-confusing
signal crafted by PGD is *not* removed, and the clone remains below threshold.

---

## 6. Scope and Non-Goals

- **In scope:** mel-domain diffusion purification (DiffPure-class), black-box
  cloning models, WavLM-based verification.
- **Out of scope (future work):** purifiers that operate in a *different*
  representation than the one whose Jacobian we analyse — notably
  EnCodec-latent-space purifiers such as VocalBridge (2026), whose Jacobian
  differs from the mel-space DiffPure Jacobian and would require extending the
  null-space projection to that latent space.
- **Assumption on ε:** protection is imperceptible only within the stated L∞
  budget; larger budgets trade audio quality for stronger confusion.
