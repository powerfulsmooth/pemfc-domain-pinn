# References and attribution

Machine-readable entries: [references.bib](references.bib).

## Physical context

1. Jaeyoo Choi, Yohan Cha, Jihoon Kong, Neil Vaz, Jaeseung Lee, Sang-Bum Ma,
   Jin-Hyuk Kim, Seung Woo Lee, Seung Soon Jang and Hyunchul Ju.
   **Multi-Variate Optimization of Polymer Electrolyte Membrane Fuel Cells in
   Consideration of Effects of GDL Compression and Intrusion.**
   Journal of The Electrochemical Society 169, 014511 (2022).
   [DOI](https://doi.org/10.1149/1945-7111/ac492f).
   Checked: original local PDF title/first article pages. Physical context;
   these published results are not the later PINN study.

2. Jaeyoo Choi, Eunsoo Kim, Yohan Cha, Masoomeh Ghasemi and Hyunchul Ju.
   **Probing the influence of nonuniform Pt particle size distribution using a
   full three-dimensional, multiscale, multiphase polymer electrolyte membrane
   fuel cell model.** Electrochimica Acta 405, 139811 (2022).
   [DOI](https://doi.org/10.1016/j.electacta.2021.139811).
   Checked: original local PDF metadata; model details examined during design.
   Multiscale catalyst context, not a newly reproduced particle-distribution result.

## PINN and diagnostic background

3. Maziar Raissi, Paris Perdikaris and George Em Karniadakis.
   **Physics-informed neural networks: A deep learning framework for solving
   forward and inverse problems involving nonlinear partial differential equations.**
   Journal of Computational Physics 378, 686–707 (2019).
   [DOI](https://doi.org/10.1016/j.jcp.2018.10.045). Foundational context.

4. Sifan Wang, Yujun Teng and Paris Perdikaris.
   **Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed
   Neural Networks.** SIAM J. Sci. Comput. 43(5), A3055–A3081 (2021).
   [DOI](https://doi.org/10.1137/20M1318043).
   Examined: preprint §2.5, Algorithm 1. Our fixed B weights do not reproduce
   that adaptive gradient-statistics algorithm.

5. Sifan Wang, Xinling Yu and Paris Perdikaris.
   **When and why PINNs fail to train: A neural tangent kernel perspective.**
   Journal of Computational Physics 449, 110768 (2022).
   [DOI](https://doi.org/10.1016/j.jcp.2021.110768).
   Examined: local PDF theory/numerics. Linear/infinite-width theory does not
   establish convergence of this finite nonlinear PEMFC network.

6. Sifan Wang, Shyam Sankaran, Hanwen Wang and Paris Perdikaris.
   **An Expert's Guide to Training Physics-informed Neural Networks.**
   [arXiv:2308.08468v1](https://arxiv.org/abs/2308.08468v1), version consulted (2023).
   Examined: Algorithm 1, nondimensionalization and balancing sections.

## Inspiration and deferred method

7. Hanwen Wang and Paris Perdikaris.
   **Latent-MoE: Domain-Aware Mixture-of-Experts for PDEs with Multi-Regime Physics.**
   [arXiv:2609.07814v1](https://arxiv.org/abs/2609.07814v1), preprint, 7 September 2026.
   Inspired regime-wise decomposition; the current fixed-routing three-expert
   model is not that paper's interleaved shared-backbone implementation.

8. Jiachen Yao, Chang Su, Zhongkai Hao, Songming Liu, Hang Su and Jun Zhu.
   **MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training
   of Physics-informed Neural Networks.** ICML 2023.
   [Official paper](https://proceedings.mlr.press/v202/yao23c.html).
   Examined: §4.3, Algorithm 2. Discussed/postponed; not implemented in the trials.

No paper figure/PDF/code is rehosted. Paper results, project observations and
proposed methods are separate; reading a method does not establish reproduction.
