# Training protocol

Original objective: L = L_data + α (L_PDE + L_BC + L_interface).
α is not the Adam learning rate. The original hierarchical reduction had
226 terms: 71 PDE supports, 11 data, 72 boundary and 72 interface terms.
Vector momentum retains its component mean. Averaging all 226 equally changes
the objective. Ten PDE families cover mass, momentum, H2, O2, N2, bulk water,
membrane water, proton transport, electron transport and energy.

## Separate lineages

- Historical branch: 200 updates with data and N2-related physics → 2 joint
  updates → 40 more, total 242. Last-block α=.003… .042; starting fixed probe
  α=.002. All 40 steps clipped; zero numerical rejections. The numerical guard
  was not a line search and did not require objective reduction.
- A/B: separate data-only 200-update control, same model and stored Adam state
  for both branches, matched plans, LR=1e-4 and α=.001… .010, 10 updates each.
  A used original physics coefficients; B used fixed calibrated downweights.
  B is not MultiAdam, GradNorm, ReLoBRaLo or adaptive learning-rate annealing.

Historical data-stage LR=1e-3; joint LR=1e-4; Adam β=(.9,.999), ε=1e-8,
global raw-gradient clip=1. Two hundred optimizer updates are not 200 full epochs.
These settings are descriptive, not universal recommended hyperparameters.

Diagnostics separated changing-sample/ramping objectives from fixed probes and
compared field values, gradients and interfaces. A/B reports CPU execution;
main training artifacts report CUDA. Their wall times are not a hardware-neutral
benchmark. Model execution occurred in Colab; NumPy analysis used returned exports.

The public trainer accepts already reduced category losses. It does not recreate
native sampling, private property gates or original control-volume operators.
A lower-LR restart and CFD-prescribed channel flow remain unexecuted proposals/designs.
