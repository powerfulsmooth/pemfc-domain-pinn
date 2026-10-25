# Interim report

## Outcome

The model executes but does not accurately recover all important spatial fields.
Residual reduction and field reconstruction diverged. Gradient diagnostics and
short controls exposed a data-versus-conservation tradeoff.

## Historical main branch

Total 202 → 242 updates. Fixed PDE loss: 1,985,743 → 6,580.52 after 30 more
updates → 40,221.23 after 40. The last change is a **6.11× fixed-probe rebound**;
variation was not solely caused by different samples. End PDE loss was still
97.97% below start. Validation normalized MSE .01324024 → .01340710 (+1.26%).
Membrane potential RMS 14.1639 → 14.1549 mV: nearly flat. Cathode-channel native
water CV contributed 98.84% of final PDE loss. All 40 steps clipped.
Plotted category losses are unweighted by α; displayed training objectives have
varying α/plans and cannot be treated as one fixed objective curve.

## Separate data-only control, 200 updates

Validation normalized MSE .00878934. Selected CFD-relative L2 errors:

| Field | Validation error |
|---|---:|
| Membrane proton potential | 29.68% |
| Membrane hydration | 9.82% |
| CCL oxygen mass fraction | 44.00% |
| Cathode-channel transported water | 3.52% |
| Cathode-channel velocity vector | 50.86% |
| Cathode-channel gauge pressure | 1.78% |

Membrane potential and CCL oxygen variation amplitudes were .4615 and .4618 of
CFD respectively. This baseline was already underfit. These are same-case
validation metrics, not unseen-case tests or normalized-loss percentages.

## Matched A/B, 10 updates each

- A, original coefficients: validation MSE +4.69%, original water CV squared
  residual −78.37%, N2 CL–membrane flux mismatch RMS −53.51%.
- B, fixed downweights: validation MSE −0.79%, water CV squared residual −21.77%,
  N2 flux mismatch RMS −2.01%.

A clipped 10/10, B 0/10. Validation field/region RMS improved for A29/80 and
B52/80 pairs. Counts do not express physical importance or significance.
B reconstructed-current correlation was about .169 despite a close mean.
This is reconstructed current, not a saved Fluent face quantity.

B's positive but very small coefficients preserve data while weakening some
physical transfers. Neither branch is selected as the final method.

## Supported and unresolved causes

Supported: selected fields underfit before joint learning; very unequal task
gradients; observable data/residual tradeoffs; clipping alone insufficient;
value/derivative/amplitude diagnostics needed independently.

Unresolved: unique cause of every error, network capacity as primary cause,
necessity of a Transformer, guaranteed benefit of a lower LR, complete equivalence
of native interface/discrete treatments. Extra training does not resolve these.

## Next design, not results

Preserve the baseline; supply CFD u,v,w,p in both channels in a separate design.
Resolve interface mass-carrier/density consistency before implementation.
No prescribed-flow, MultiAdam or Transformer trial is reported here.

Sources: selected verified returned Colab aggregates in [summary.json](summary.json)
and [fixed-probe table](main_branch_fixed_probe.csv).
[Metric definitions](../../docs/evaluation.md) · [Date provenance](../../docs/history.md).
Raw arrays and contours remain private.
