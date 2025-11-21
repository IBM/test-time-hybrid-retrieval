import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Global style tweaks for a cleaner, paper-ready look
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

METRICS = [
    "biomedical_lectures_v2",
    "economics_reports_v2",
    "esg_reports_human_labeled_v2",
    "esg_reports_v2",
    "average",
]

# Marker styles for different learning rates (avoid '*' because T=0 uses it)
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "H", "p"]

def format_sci(x):
    """Format floats in scientific notation (1e-4 style)."""
    try:
        return f"{float(x):.0e}"
    except Exception:
        return x

def plot_metric(df, metric, plot_title=None, out_dir=None):
    df_metric = df.copy()

    # Ensure numeric
    df_metric["n_steps"] = pd.to_numeric(df_metric["n_steps"], errors="coerce")
    df_metric[metric] = pd.to_numeric(df_metric[metric], errors="coerce")
    df_metric["lr"] = pd.to_numeric(df_metric["lr"], errors="coerce")

    # Drop invalid rows
    df_metric = df_metric.dropna(subset=["n_steps", metric, "lr"])
    if df_metric.empty:
        print(f"[WARN] No valid data for metric '{metric}', skipping.")
        return

    # Aggregate over (lr, n_steps): one point per combination
    df_agg = (
        df_metric
        .groupby(["lr", "n_steps"], as_index=False)[metric]
        .mean()
    )

    # Unique learning rates (this is the ONLY grouping)
    lrs = sorted(df_agg["lr"].unique())
    cmap = plt.get_cmap("tab20")

    # Map each learning rate to a display label and marker
    lr_label = {lr: format_sci(lr) for lr in lrs}
    marker_map = {
        lr: MARKERS[i % len(MARKERS)]
        for i, lr in enumerate(lrs)
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Emphasized T = 0 point (shared across lrs, from aggregated data)
    df_zero = df_agg[df_agg["n_steps"] == 0]
    if not df_zero.empty:
        val = df_zero.iloc[0][metric]
        ax.scatter(
            [0],
            [val],
            s=300,
            color="black",
            marker="*",
            edgecolor="white",
            linewidth=0.8,
            label="T = 0",
            zorder=5,
        )

    # Points grouped ONLY by learning rate
    for i, lr in enumerate(lrs):
        sub = df_agg[(df_agg["lr"] == lr) & (df_agg["n_steps"] != 0)]
        if sub.empty:
            continue
        sub = sub.sort_values("n_steps")
        ax.scatter(
            sub["n_steps"],
            sub[metric],
            s=25,
            color=cmap(i % 20),
            marker=marker_map[lr],
            edgecolor="black",
            linewidth=0.4,
            label=lr_label[lr],
            alpha=0.9,
        )

    ax.set_xlabel("T (number of steps)")
    ax.set_ylabel(f"Performance ({metric})")

    # Use provided plot title if given, otherwise default
    if plot_title is not None:
        ax.set_title(plot_title)
    else:
        ax.set_title(metric.replace("_v2", "").replace("_", " ").title())

    # Y axis formatting
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)

    ymin, ymax = df_agg[metric].min(), df_agg[metric].max()
    margin = (ymax - ymin) * 0.05 if ymax > ymin else 0.01
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(75)

    # Clean axes and grid
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4)

    # Legend (deduplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="learning rate",
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=1.5,
        borderaxespad=0.8,
        title_fontsize=9,
    )

    fig.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{metric}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    else:
        plt.show()


def main(csv_path, out_dir=None, plot_title=None):
    df = pd.read_csv(csv_path)
    for metric in METRICS:
        if metric not in df.columns:
            print(f"[WARN] Column '{metric}' not in CSV, skipping.")
            continue
        plot_metric(df, metric, plot_title=plot_title, out_dir=out_dir)


if __name__ == "__main__":
    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_title = sys.argv[3] if len(sys.argv) > 3 else None
    main(csv_path, out_dir, plot_title)
