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
scoring        = df["scoring"].values

# -------------------------------------------------
# Colors & hatches (matching your screenshot)
# -------------------------------------------------
colors = {
    "PrimaryQuery":  "#6aaed6",  # blue
    "PrimaryIndex":  "#08519c",  # dark blue
    "CompQuery":     "#fdae6b",  # orange
    "CompIndex":     "#f16913",  # darker orange
    "Optimization":  "#de2d26",  # red
    "Scoring":       "#9e9ac8",  # purple
}

hatches = {
    "PrimaryQuery":  "///",
    "PrimaryIndex":  "\\\\\\",
    "CompQuery":     "++",
    "CompIndex":     "**",
    "Optimization":  "|||",
    "Scoring":       "oo",
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
    ("Primary query encoding",      primary_query, colors["PrimaryQuery"], hatches["PrimaryQuery"]),
    ("Primary index search",        primary_index, colors["PrimaryIndex"], hatches["PrimaryIndex"]),
    ("Complementary query encoding", comp_query,   colors["CompQuery"],    hatches["CompQuery"]),
    ("Complementary index search",   comp_index,   colors["CompIndex"],    hatches["CompIndex"]),
    ("Test-time optimization",       optimization, colors["Optimization"], hatches["Optimization"]),
    ("Scoring",                      scoring,      colors["Scoring"],      hatches["Scoring"]),
]

fig, ax = plt.subplots(figsize=(8, 1.8 * len(variants)))

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

# Add total latency labels
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

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
)

plt.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
