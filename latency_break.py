import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv("latency_breakdown.csv")

variants = df["run_id"].values

primary_query  = df["primary_query_encoding_latency"].values
primary_index  = df["primary_index_search_latency"].values
comp_query     = df["complementary_query_encoding_latency"].values
comp_index     = df["complementary_index_search_latency"].values
optimization   = df["test_time_optimization"].values

# -------------------------------------------------
# Colors & hatches
# -------------------------------------------------
colors = {
    "PrimaryQuery":  "#6aaed6",
    "PrimaryIndex":  "#08519c",
    "CompQuery":     "#fdae6b",
    "CompIndex":     "#f16913",
    "Optimization":  "#de2d26",
}

hatches = {
    "PrimaryQuery":  "///",
    "PrimaryIndex":  "\\\\\\",
    "CompQuery":     "++",
    "CompIndex":     "**",
    "Optimization":  "|||",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.edgecolor": "#333333",
})

# -------------------------------------------------
# Plot
# -------------------------------------------------
y_pos = np.arange(len(variants))
height = 0.35

components = [
    ("Primary query encoding",       primary_query, colors["PrimaryQuery"],  hatches["PrimaryQuery"]),
    ("Primary index search",         primary_index, colors["PrimaryIndex"],  hatches["PrimaryIndex"]),
    ("Complementary query encoding", comp_query,    colors["CompQuery"],     hatches["CompQuery"]),
    ("Complementary index search",   comp_index,    colors["CompIndex"],     hatches["CompIndex"]),
    ("Test-time optimization",       optimization,  colors["Optimization"],  hatches["Optimization"]),
]

# Figure with 2 rows: top = plot, bottom = legend
fig = plt.figure(figsize=(8, 1.8 * len(variants) + 1.5))
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1])

ax = fig.add_subplot(gs[0])
ax_legend = fig.add_subplot(gs[1])

left = np.zeros(len(variants))

for label, values, color, hatch in components:
    ax.barh(
        y_pos,
        values,
        left=left,
        height=height,
        color=color,
        edgecolor="black",
        linewidth=0.7,
        hatch=hatch,
        label=label,
    )
    left += values

totals = left

for y, total in zip(y_pos, totals):
    ax.text(
        total + totals.max() * 0.01,
        y,
        f"{int(total)}ms",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(variants)
ax.invert_yaxis()

ax.set_xlabel("Latency (ms)")
ax.set_xlim(0, totals.max() * 1.15)

# -------------------------------------------------
# Legend in its own axis at the bottom
# -------------------------------------------------
ax_legend.axis("off")
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(
    handles,
    labels,
    loc="center",
    ncol=3,
    frameon=True,
)

fig.tight_layout()

fig.savefig("latency_break.png", dpi=300, bbox_inches="tight")
plt.close(fig)
