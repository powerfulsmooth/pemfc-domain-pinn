# Physical scope

High-level physical context: [compression/intrusion paper](https://doi.org/10.1149/1945-7111/ac492f)
and [multiscale Pt paper](https://doi.org/10.1016/j.electacta.2021.139811).
This description is not a release of private constitutive/source formulas.

| Family | Main states investigated | Equation roles |
|---|---|---|
| Channels | u,v,w,p,T,reactant,W,N2 | Laminar mass/momentum, convection–diffusion, energy |
| GDL | Flow, species, water, T, electron potential | Porous resistance, transport, conduction |
| CL | Flow, species, water, T, both potentials, N2 | Transport and volumetric electrochemical sources |
| Membrane | T,λ,proton potential,N2 under the frozen contract | Hydration, proton transport, thermal coupling |
| BP | T, electron potential | Heat and electron conduction |

A generic scalar balance is div(ρ u q − Γq grad(q)) − Sq = 0.
Its transported variable, coefficient, source and phase interpretation must be
defined consistently. Water-variable substitution is not a notation-only change.
The private implementation includes Kirchhoff-transformed water treatment and
selected native control-volume supports. The synthetic diffusion demo does not.

Constitutive laws, variable transforms and source formulas feed PDE residuals;
they are not all independent loss equations. Under-relaxation and source
linearization derivatives are iteration tools, not additional physical laws.

Interface laws follow the actual transport path and sources/contact model.
Conservation does not generally imply equal gradients or neighboring cell-center
values. Fixing velocity does not fix mass transport when density remains learned.
Interface equivalence to every native Fluent discrete treatment is not established.
Historical results retain the original coupled-flow model.
