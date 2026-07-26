# Handoff: Stage-2 Path Arbiter Direction Review

Date: 2026-06-27

## Context

This line tried to improve weak-packet Stage-2 decoding under:

```text
Savaux over-sampled Stage-1 candidates
-> phase-line / Viterbi Stage-2
-> compare against the earlier one-order phase penalty DP baseline
```

The user concern is valid: the recent work drifted toward conservative fallback / arbitration, while the real goal was to beat the older pure/one-order VTB by allowing better whole-packet path search, especially after Stage-1 energy improves.

## Main Code Changes

New version-management files under `weak_decoder/phase_line/`:

- `stage2_types.py`: Stage-2 variant names and decision dataclass.
- `stage2_profiles.py`: aggressive/rewrite config profiles.
- `stage2_decision.py`: packet confidence stats and aggressive arbitration.
- `stage2_variants.py`: dispatch for stable island, aggressive island, adaptive island, rewrite island.
- `stage2_arbiter.py`: first draft `v1_risk_arbiter`.

Existing touched files:

- `configs.py`: added adaptive/aggressive/rewrite/risk-arbiter parameters.
- `savaux_stage1.py`: added wrapper entry points for adaptive/rewrite/risk arbiter.
- `__init__.py`: exported new APIs.
- `_eval/diagnose_stage2_v2_bidirectional.py`: expanded metrics: SER, rescue/break/net, CRC valid, path change counts, CRC/risk arbiter.

## Key Results

Baseline on `0_0_0_10_14_16`, 165 packets:

```text
Savaux hard SER      0.26649
v1 one-order VTB SER 0.21420   rescue/break/net = 395/93/+302
stable island SER    0.25281   rescue/break/net = 96/17/+79
adaptive island SER  0.25160   rescue/break/net = 103/17/+86
```

Three datasets, 420 packets:

```text
Savaux hard SER      0.19605
v1 one-order VTB SER 0.18102   rescue/break/net = 491/270/+221
stable island SER    0.18912   rescue/break/net = 147/45/+102
adaptive island SER  0.18857   rescue/break/net = 158/48/+110
```

So island/adaptive variants are safer than raw VTB, but they do **not** beat the earlier v1 one-order VTB. This is the main reason the direction feels off.

## Experiments That Did Not Work

### Harder/softer top1 protection

Reliable top1 soft bonus reduced break, but lost more rescue:

```text
v1 baseline SER       0.21420  rescue/break/net = 395/93/+302
bonus 0.02 SER        0.21524  rescue/break/net = 377/81/+296
bonus 0.04 SER        0.21610  rescue/break/net = 363/72/+291
bonus 0.06 SER        0.21749  rescue/break/net = 349/66/+283
```

Conclusion: protecting top1 moves back toward conservative Stage-1, not toward better decoding.

### Rewrite/aggressive island

The permissive rewrite profile allowed large path changes, but collapsed:

```text
rewrite SER           0.39740
rewrite rescue/break  196/952
```

Conclusion: simply widening Top-L and increasing phase/profile authority makes the path drift badly.

### CRC arbiter

CRC-aware selection is production-available and improves CRC count, but not enough:

```text
v1 CRC valid          10 / 165
adaptive CRC valid     5 / 165
CRC union             10 / 165
crc-arb SER           0.24848
```

Most CRC wins already come from v1. CRC arbiter is useful for packet validity reporting, but not the core SER solution.

## Most Useful Observation

Offline rule replay showed:

```text
if v1.trajectory_score <= 0.8:
    use v2 bidirectional
else:
    use v1
```

On the 165-packet set this gives about:

```text
SER 0.20779
```

This beats v1 on that set, using only decoder-visible features. It has **not** yet been fully implemented/validated as a production path across the 3-dataset benchmark.

## Recommendation For Next Agent

Do not keep tuning island/adaptive as the main path. It is too conservative and loses the old v1 rescue advantage.

The most promising next step is:

```text
base path = v1 one-order VTB
risk gate = v1 internal trajectory/phase quality
fallback = v2 bidirectional or another stable path
```

Then validate against:

```text
v1 one-order VTB, not just Savaux hard
SER
rescue/break/net
CRC valid
low-SNR packet wins
3-dataset stability
```

Avoid using GT-derived gates. The rule may be tuned with GT offline, but final features must be decoder-visible: `trajectory_score`, `mean_phase_score`, `change_count`, `anchor_ratio`, `CRC valid`, etc.

## Current Caution

There are many uncommitted/generated experiment outputs in `_eval/`. Do not clean or revert blindly. The codebase is dirty, and some files touched before this handoff were already modified.
