# Metrics

Volume-weighted RMS = sqrt(sum(V * (prediction − reference)^2) / sum(V)).
Relative L2 [%] = 100 * sqrt(sum(V * error^2) / sum(V * reference^2)).

This is not MAPE, error divided by reference mean, or normalized loss times 100.
Zero reference norm means undefined; no denominator floor is used. Tiny
reference velocities can produce large percentages. Absolute-Kelvin temperature
percentages can hide inaccurate variation. Pressure and potential retain their
original reference without shifting.

Equal-tag normalized data MSE uses active-field scales, volume weighting within
a support and equal material-tag averaging. It is not overall physical accuracy.
Squared water CV residual and N2 flux mismatch RMS have different definitions;
their percentage reductions must not be interchanged.

Validation cells are from the same case; final locked test labels were unused
for these diagnostics. Spatial holdout is not unseen-geometry/condition validation.
Reference CFD is numerical, not experimentally established ground truth.

Membrane current was reconstructed from conductivity and 3D potential gradients.
CFD gradients used a same-material least-squares reconstruction, not saved Fluent
face-current data. Mean agreement alone does not show local distribution accuracy.

Private XY/YZ/ZX contour comparisons used common scales; those spatial arrays and
figures remain withheld. Public figures show selected aggregate results only.
