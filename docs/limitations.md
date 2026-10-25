# Limitations

- Accurate/converged full-field PEMFC prediction has not been established.
- One geometry/condition; no demonstrated operator learning or generalization.
- Low residual, matching means and good field values do not establish good derivatives.
- A/B has one start and 10 updates per branch, without repeated-seed significance.
- Clipping does not equalize task gradients; norms are not Adam update ratios.
- Complete equivalence with native Fluent interface/discrete operators is unresolved.
- CFD references and reconstructed gradients are numerical, not experimental truth.
- Exact historical reproduction requires withheld data and case-specific code.
- Public Torch code was syntax-checked but not locally executed during packaging.
- No benefit has been demonstrated for the proposed lower LR, prescribed channel
  flow, MultiAdam or Transformer changes.
