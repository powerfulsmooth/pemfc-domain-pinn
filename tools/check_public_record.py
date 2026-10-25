"""Standard-library verification of public aggregates; no Torch or private files."""
import csv
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results/interim/summary.json").read_text(encoding="utf8"))
    with (root / "results/interim/main_branch_fixed_probe.csv").open(encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["total"]) for r in rows] == [202, 212, 222, 232, 242]
    assert math.isclose(100*(1-float(rows[-1]["PDE"])/float(rows[0]["PDE"])), 97.9745, abs_tol=.0001)
    assert math.isclose(float(rows[-1]["PDE"])/float(rows[-2]["PDE"]), 6.1122, abs_tol=.001)
    observations = data["paired_trial"]["observations"]
    baseline = observations["start"]
    a, b = observations["A_final"], observations["B_final"]
    key = "validation_equal_tag_normalized_MSE"
    assert 4.68 < 100*(a[key]/baseline[key]-1) < 4.70
    assert -.80 < 100*(b[key]/baseline[key]-1) < -.78
    assert data["model"]["trainable_parameters"] == sum(
        sum((x+1)*y for x, y in zip(widths, widths[1:]))
        for widths in ((6,48,48,48,10), (9,48,48,48,9), (5,24,24,24,2), (32,16,10)))
    print("Public aggregate consistency checks passed (no Torch execution).")


if __name__ == "__main__":
    main()
