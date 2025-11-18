import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python gqr_group_by_lr_steps.py <path_to_csv>")
        return

    path = sys.argv[1]
    df = pd.read_csv(path)

    # Ensure numeric
    df["average"] = pd.to_numeric(df["average"], errors="coerce")
    df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    df["n_steps"] = pd.to_numeric(df["n_steps"], errors="coerce")

    # Keep only the GQR method
    gqr = df[df["method"] == "GQR"]

    # Group by learning rate and num steps
    grouped = (
        gqr.groupby(["lr", "n_steps"])["average"]
        .mean()
        .reset_index()
        .sort_values(["lr", "n_steps"])
    )

    print("\nGQR mean performance by learning rate and n_steps:\n")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
