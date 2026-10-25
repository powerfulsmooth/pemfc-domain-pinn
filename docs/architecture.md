# Network architecture

| Expert | Regions | Widths | Parameters |
|---|---|---|---:|
| MEA | ACL, membrane, CCL | 6 → 48 → 48 → 48 → 10 | 5,530 |
| GC/GDL | Both channels and compressed/uncompressed GDLs | 9 → 48 → 48 → 48 → 9 | 5,625 |
| BP | Both bipolar plates | 5 → 24 → 24 → 24 → 2 | 1,394 |
| Common readout | Pointwise, all tags | 32 → 16 → 10 | 698 |

Total 13,247; tanh hidden activations; float64. Expert inputs are three locally
normalized coordinates plus a local one-hot material tag: 3+3, 3+6 or 3+2.

Twenty-one slots are concatenated expert-output positions (10+9+2), not spatial
points. Only the assigned expert fills its block; the other blocks are zero.
Eleven global material tags complete the 32 readout inputs. The head produces
a correction to the mapped expert outputs. A material mask suppresses inactive
slots. The private decoder and train-derived offsets follow this step.

| Shared slots | Private physical interpretation |
|---|---|
| 0,1,2 | Three velocity components |
| 3 | Gauge pressure with stored reference |
| 4 | Temperature |
| 5 | Side-dependent H2/O2 reactant mass fraction |
| 6 | Transported water variable or membrane hydration λ |
| 7,8 | Proton and electron potentials |
| 9 | N2 mass fraction |

W is not automatically gas water-vapor mass fraction; λ is a distinct membrane
state. Saturation, conductivity and reaction rates are derived quantities.
Eleven tags preserve side/compression identities within three expert families.

The public network preserves topology and active masks but returns dimensionless
slots: private normalization, offsets and physical transforms are withheld.
Regime-wise decomposition was inspired by [Latent-MoE](https://arxiv.org/abs/2609.07814v1).
This fixed-routing model does not reproduce its interleaved backbone, compact
router or NTK guarantees. The readout does not access other spatial points.
