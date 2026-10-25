"""Regenerate public figures from selected aggregate tables, not spatial CFD data.

Optional dependency: matplotlib. Outputs are static scientific figures.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    with (ROOT / "results/interim/main_branch_fixed_probe.csv").open(encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    x = [int(r["total"])-202 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for key, label, color in (("PDE", "PDE", "#287b8e"), ("interface", "Interface", "#cc7832")):
        values = [float(r[key])/float(rows[0][key]) for r in rows]
        axes[0].semilogy(x, values, "o-", lw=2, color=color, label=label)
    axes[0].set(ylabel="Fixed-probe loss / start", xlabel="Additional joint updates",
                title="Residual reduction and rebound")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    values = [100*(float(r["validation"])/float(rows[0]["validation"])-1) for r in rows]
    axes[1].plot(x, values, "o-", color="#ab477a", lw=2)
    axes[1].axhline(0, color="gray", lw=.8)
    axes[1].set(ylabel="Validation MSE change from start [%]",
                xlabel="Additional joint updates", title="Field fit did not improve")
    axes[1].grid(alpha=.2)
    fig.suptitle("Historical main branch: total updates 202 to 242", y=.99, fontsize=14)
    fig.text(.5, .012, "Unweighted category losses on fixed probes; alpha and training sample plans varied.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(FIG / "main_branch_progress.png", dpi=180)
    plt.close(fig)

    summary = json.loads((ROOT / "results/interim/summary.json").read_text(encoding="utf8"))
    obs = summary["paired_trial"]["observations"]
    metrics = [("validation_equal_tag_normalized_MSE", "Validation\nnormalized MSE"),
               ("CGC_all1000_original_water_loss", "Water CV\nsquared residual"),
               ("CCL_MEM_N2_flux_jump_RMS_kg_m2_s", "N2 interface\nflux RMS")]
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    for offset, name, label, color in ((-.18,"A_final","A: original coefficients","#287b8e"),
                                      (.18,"B_final","B: fixed downweights","#cc7832")):
        changes = [100*(obs[name][key]/obs["start"][key]-1) for key,_ in metrics]
        bars = ax.bar([i+offset for i in range(3)], changes, .34, label=label, color=color)
        for bar, value in zip(bars, changes):
            ax.text(bar.get_x()+bar.get_width()/2, value+(1.8 if value >= 0 else -1.8),
                    f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xticks(range(3), [label for _,label in metrics])
    ax.set(ylabel="Change from the common start [%]", ylim=(-95,17),
           title="Matched 10-update A/B: preserving fit weakens some physical progress")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=.15)
    fig.text(.5,.015,"Smaller is better for each displayed metric. Squared residual and RMS are distinct measures.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(FIG / "paired_tradeoff.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,8.6))
    ax.set(xlim=(0,12), ylim=(0,8.6))
    ax.axis("off")
    def box(x,y,w,h,text,color="#eaf2f6",size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",
                                   facecolor=color,edgecolor="#476477",linewidth=1.3))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,linespacing=1.5)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,color="#476477",lw=1.3))
    ax.text(6,8.3,"3D coordinate PINN: three experts + one pointwise readout",ha="center",fontsize=16,weight="bold")
    box(2.6,7.15,6.8,.7,"Input: normalized (x, y, z) + material tag")
    arrow((6,7.07),(6,6.95))
    ax.text(6,6.67,"Fixed routing: evaluate only the assigned expert at each point",ha="center",fontsize=10)
    box(.3,4.9,3.5,1.35,"MEA expert\n6 → 48 → 48 → 48 → 10\n5,530 parameters", "#e3f1ed")
    box(4.25,4.9,3.5,1.35,"GC / GDL expert\n9 → 48 → 48 → 48 → 9\n5,625 parameters", "#e4edf9")
    box(8.2,4.9,3.5,1.35,"BP expert\n5 → 24 → 24 → 24 → 2\n1,394 parameters", "#f5eadc")
    for cx in (2.05,6,9.95):
        arrow((6,6.48),(cx,6.32))
        arrow((cx,4.81),(cx,4.38))
    box(.3,3.45,11.4,.82,"Aligned 21 expert slots (10 + 9 + 2) + 11 global material tags = 32\nOnly the routed expert fills its block; other blocks are zero",size=11)
    arrow((6,3.37),(6,3.0))
    box(2.6,2.12,6.8,.75,"Common residual readout: 32 → 16 → 10  |  698 parameters")
    arrow((6,2.03),(6,1.68))
    box(1, .72,10,.82,"Mapped expert output + readout correction → active mask → physical decoder\nPrivate decoder/offsets withheld; public code returns dimensionless slots", "#f2edf6", 10.5)
    ax.text(6,.26,"Total 13,247 parameters  •  tanh / float64  •  no attention between spatial samples",ha="center",fontsize=11,color="#334d60")
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "architecture.svg", bbox_inches="tight")
    fig.savefig(FIG / "architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Four public aggregate/architecture figures regenerated.")


if __name__ == "__main__":
    main()
