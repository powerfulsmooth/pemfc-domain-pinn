# Private experiment versus independently authored public code

| Public component | Retained | Difference |
|---|---|---|
| model.py | Expert dimensions, routing, packing, residual head, active masks | No private bounds, offsets or physical decoder; dimensionless output |
| operators.py | Independent-point 3D autograd | Generic scalar diffusion only |
| training.py | Category objective, fixed-parameter chunk accumulation, clipping, Adam, nonfinite rollback | No private sampling/property/outlet gates or native water |
| synthetic.py | Exercises data/PDE/BC/interface learning | Analytic 11 equal slabs, no fuel-cell physics |
| train_synthetic.py | Warmup, ramp and resumable state | Not historical training orchestration |
| results/interim | Selected verified aggregate results | No arrays/contours/checkpoints |

These new public files are not the full source used to generate the PEMFC results.
A private decoder/material/geometry implementation is necessary to reproduce the
historical experiment. Synthetic success would demonstrate software only.
