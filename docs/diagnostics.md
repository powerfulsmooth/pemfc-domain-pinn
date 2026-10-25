# What the diagnostics found

This is a selected aggregate account of zero-update diagnostics and short controls,
not a new training run or a proof of one universal failure mechanism.

| Observation | Evidence | Interpretation |
|---|---|---|
| Different initial training paths learn different fields | Membrane potential RMS: data-only 200 = 10.820 mV; early data+N2 200 = 14.166 mV; main 242 = 14.155 mV | Stagnation was visible before the final 40 updates; not solely a late-ramp problem |
| The supervised control still underfits | Potential and CCL oxygen amplitude ratios about .462 | More than loss balancing may eventually be needed; the control is not a solved reference |
| Early physics substantially changes optimization scale | Global gradient-norm median .03078 with data only versus 21.80 with early N2; clipping 0/200 versus 200/200 | A numerically small physics coefficient does not guarantee small gradient influence |
| Water CV dominates the late direction | At total 242 and alpha=.042, GC/GDL water-CV gradient norm about 827,996 versus data .04101 within that same expert | Extremely unequal optimization signals; not automatically an Adam update ratio |
| MEA has its own imbalance | N2 interface gradient norm about 173.002 versus MEA data .008480 | Water is not the direct cause in every module |
| Values and derivatives need separate checks | Main-branch membrane x-potential-gradient RMS about 11.83 V/m versus reconstructed CFD 1,345.8 V/m | A seemingly modest potential RMS can coexist with severely inaccurate transport gradients |
| Weight reduction produces a tradeoff | B preserves validation fit but N2 flux mismatch RMS falls only 2.01% in 10 steps | Tiny positive weights keep equations present but can weaken physical progress |

Gradient measurements are Euclidean parameter gradients at a fixed checkpoint,
before global clipping and after the stated alpha. They are not actual Adam
parameter displacements, percentages of contribution, or guaranteed future changes.
In particular, norm ratios do not alone prove directional opposition.

The scalar categories were recombined and compared with the direct objective
gradient; agreement in the private audit was near floating-point precision.
That is evidence for the decomposition, not evidence of accurate physical fields.

The data-only and early-N2 controls share the intended seed/data-sampling design.
One seed/case still limits causal generality. Intermediate fields inside the first
200 steps were not available to determine whether spatial variation was learned
and then lost or never formed adequately.

These observations motivate first simplifying the learning task and resolving
transport coupling. They do not yet justify a Transformer, additional experts,
MultiAdam, or deletion of a conservation law as the proven solution.
