#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# === CONFIG ===
base_dir = Path("/home/lknoche/maby")
output_dir = base_dir / "output"
figures_dir = base_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Evaluations
evaluations = [
    ("Vph1_GFP", "fixed", "2025-05-15_train_Vph1_GFP", "model_19.pt", "fixed_0.01"),
    ("Htb2_sfGFP", "fixed", "2025-05-26_train_Htb2_sfGFP_resume", "model_39.pt", "fixed_0.1"),
    ("Htb2_sfGFP", "otsu", "2025-05-26_train_Htb2_sfGFP_resume", "model_39.pt", "otsu"),
]

def generate_violin(which="100"):
    print(f"\n📊 Generating violin plots for: {'First 100' if which == '100' else 'All Timepoints'}")
    rows = []
    for gfp, method, experiment, _, method_tag in evaluations:
        label = "Nucleus" if "Htb2" in gfp else "Vacuole"
        comp = f"{label} ({method.capitalize()})"

        # Define correct filenames
        if gfp == "Vph1_GFP":
            filename = f"test_results_{method_tag}_0_100.csv"
        elif gfp == "Htb2_sfGFP" and method == "fixed":
            filename = f"test_results_fixed_0.1_0_100.csv" if which == "100" else f"test_results_fixed_0.1_0_277.csv"
        else:
            filename = f"test_results_{method_tag}_{'100' if which == '100' else 'all'}.csv"

        csv_path = output_dir / experiment / filename
        if not csv_path.exists():
            print(f"⚠️  Skipping {csv_path} (not found)")
            continue

        print(f"📂 Reading: {csv_path}")
        df = pd.read_csv(csv_path)

        # Rename F1 column as needed
        if gfp == "Vph1_GFP":
            df = df.rename(columns={"F1@0.01": "F1"})
        elif gfp == "Htb2_sfGFP" and method == "fixed":
            df = df.rename(columns={"F1@0.1": "F1"})
        else:
            f1_col = next((col for col in df.columns if col.startswith("F1")), None)
            df = df.rename(columns={f1_col: "F1"})

        df["Compartment"] = comp
        rows.append(df[["Similarity", "F1", "Compartment"]])

    df_all = pd.concat(rows)

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette = {
        "Nucleus (Fixed)": "#E76F51",
        "Nucleus (Otsu)": "#2A9D8F",
        "Vacuole (Fixed)": "#264653",
    }

    sns.violinplot(data=df_all, x="Compartment", y="Similarity", ax=axes[0], palette=palette)
    sns.violinplot(data=df_all, x="Compartment", y="F1", ax=axes[1], palette=palette)
    axes[0].set_title("Cosine Similarity")
    axes[1].set_title("F1 Score")
    fig.suptitle(f"Segmentation Performance ({'100 TPs' if which == '100' else 'All TPs'})")
    plt.tight_layout()

    out_path = figures_dir / f"violin_plot_{which}.png"
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")
    plt.close()

    # Save summary table
    means = df_all.groupby("Compartment").mean(numeric_only=True)
    means.to_csv(figures_dir / f"metrics_summary_{which}.csv")
    print("\n📋 Markdown Table:\n")
    print(means[["F1", "Similarity"]].to_markdown())
    print("\n📋 LaTeX Table:\n")
    print(means[["F1", "Similarity"]].to_latex(float_format="%.3f", index=True))


if __name__ == "__main__":
    generate_violin("100")
    generate_violin("all")
