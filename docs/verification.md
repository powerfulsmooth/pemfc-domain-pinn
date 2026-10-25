# Verification status

Previous private checks and new public code are different scopes.
Private returned aggregates were recomputed from Colab exports and lineage
reviewed. Private component/network/residual checks do not validate new public code.

During packaging, Python files are checked for syntax without importing Torch,
parameter count is recomputed arithmetically, public tables are checked against
verified source summaries, and content/links/figure layout are reviewed.

Public Torch tests, inference and optimizer steps are **not run locally**.
Colab runs tests before the demo: routing/masking, parameter count, manufactured
derivatives and chunked-gradient equivalence. No synthetic performance number is
reported without execution. Future reports must name environment/device/outcome.
