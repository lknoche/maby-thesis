# %%
from dataset import PositionDataset, transform
from dlmbl_unet import UNet
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import seaborn as sns

# %%
# Load the results
experiments = [
    ("2024-10-08", "Hog1_GFP", 0.4, "Cytoplasm"),
    ("2024-10-07", "Vph1_GFP", 0.2, "Vacuole"),
    ("2024-10-09", "Htb2_sfGFP", 0.2, "Nucleus"),
    ("2024-10-09", "YST_665", 0.6, "Bud Neck"),
]

base_dir = Path("/nrs/funke/adjavond/maby/data")
output_dir = Path("/nrs/funke/adjavond/maby/output")
# %%
# Load the data and model
data_dir = base_dir / "MAYBE_training_00"
metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"
test_position = 5
start_tp = 100
index = 300
device = "cuda" if torch.cuda.is_available() else "cpu"

for expt in experiments:
    start_date = expt[0]
    gfp_type = expt[1]

    print("Loading data for", gfp_type)
    test_dataset = PositionDataset(
        image_folder=data_dir / f"{gfp_type}_{test_position:03d}",
        h5_file=metadata_dir / f"{gfp_type}_{test_position:03d}.h5",
        transform=transform,
        start_tp=start_tp,
    )

    print("Loading model for", gfp_type)
    model = UNet(depth=3, in_channels=5, out_channels=5)
    model_checkpoint = torch.load(
        output_dir / f"{start_date}_train_{gfp_type}" / "best_model.pth"
    )["model"]
    model.load_state_dict(model_checkpoint)
    model = model.to(device)
    model.eval()

    print("Running inference for", gfp_type)
    with torch.inference_mode():
        x, y = test_dataset[index]
        y /= y.max()
        y_pred = model(x.unsqueeze(0).float().to(device)).squeeze(0)

    # Show the images
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        axes[0, i].imshow(x[i].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(y[i].cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")
        axes[2, i].imshow(y_pred[i].detach().cpu().numpy(), cmap="gray")
        axes[2, i].axis("off")
    axes[0, 0].set_title("Input")
    axes[1, 0].set_title("Ground Truth")
    axes[2, 0].set_title("Prediction")
    plt.suptitle(f"{gfp_type} - Position {test_position} - Index {index}")
    plt.show()


# %%
# Plot the test results
all_results = []
for train_date, gfp_type, chosen_threshold, compartment in experiments:
    experiment_dir = output_dir / f"{train_date}_train_{gfp_type}"

    results = pd.read_csv(experiment_dir / f"generalization_test_{test_position}.csv")
    results["Compartment"] = compartment

    results = results[["Similarity", "Compartment", f"F1@{chosen_threshold}"]]
    results.rename(columns={f"F1@{chosen_threshold}": "F1"}, inplace=True)
    all_results.append(results)
results = pd.concat(all_results)

# %%
# Plot the results
# One boxplot for similarities, colored by compartment
# One boxplot for F1 scores, colored by compartment
colors = ["orange", "green", "blue", "red"]
order = ["Cytoplasm", "Vacuole", "Nucleus", "Bud Neck"]
palette = dict(zip(order, colors))

sns.set_context("talk")
sns.set_style("whitegrid", {"grid.linestyle": ":"})
plt.style.use("dark_background")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))


sns.violinplot(
    data=results,
    x="Compartment",
    y="Similarity",
    ax=axes[0],
    hue="Compartment",
    palette=palette,
    order=order,
)
sns.violinplot(
    data=results,
    x="Compartment",
    y="F1",
    ax=axes[1],
    hue="Compartment",
    palette=palette,
    order=order,
)

axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
