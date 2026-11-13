"""Plot the frozen AR(1) stress-test cells from results/final_tables.md, Table H."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RHO = np.array([0.0, 0.3, 0.5, 0.7])
# Percentages in Table H, JKV calibration, tau=240, 5% test, half-strength factor for power.
SERIES = {
    "Original bootstrap": {"size": [3, 100, 100, 100], "power": [83, 82.5, 75.5, 70]},
    "AR(1), global residuals": {"size": [1, 2, 3, 4], "power": [74, 39.5, 22.5, 9]},
    "AR(1), local residuals": {"size": [7.5, 11.5, 14.5, 24.5], "power": [85.5, 55, 38, 30]},
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharex=True)
    colors = ["#b94b4b", "#315f8c", "#248879"]
    for (label, values), color in zip(SERIES.items(), colors):
        axes[0].plot(RHO, values["size"], "o-", lw=2, color=color, label=label)
        axes[1].plot(RHO, values["power"], "o-", lw=2, color=color, label=label)
    axes[0].axhline(5, color="#555555", lw=1, ls="--", label="Nominal 5%")
    axes[0].set_title("Size: no planted change")
    axes[1].set_title("Power: weak fourth factor at T/2")
    for ax in axes:
        ax.set_xlabel("Error AR(1) coefficient")
        ax.set_xticks(RHO)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rejection rate (%)")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Serial correlation stress test (JKV calibration, d=3)", fontsize=13)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    out = ROOT / "figures/serial_correlation_stress.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
