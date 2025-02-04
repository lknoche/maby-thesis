# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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

all_results = []
for train_date, gfp_type, chosen_threshold, compartment in experiments:
    experiment_dir = output_dir / f"{train_date}_train_{gfp_type}"

    results = pd.read_csv(experiment_dir / "test_results.csv")
    results["Compartment"] = compartment

    results = results[["Similarity", "Compartment", f"F1@{chosen_threshold}"]]
    # TODO stack them when we have the other compartments
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
# plt.style.use("dark_background")

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


# %%
# The below is just to see which of the thresholds I think is the most "accurate"
# representation of segmentation -- this is how I will pick the thresholding that I
# Report in the poster. I should share these details with the team.
from dataset import PositionDataset, transform

gfp_type = "Hog1_GFP"
test_position = 5
end_tp = 100

data_dir = base_dir / "MAYBE_training_00"
metadata_dir = base_dir / "2179_2024_05_08_MAYBE_training_00"

test_dataset = PositionDataset(
    image_folder=data_dir / f"{gfp_type}_{test_position:03d}",
    h5_file=metadata_dir / f"{gfp_type}_{test_position:03d}.h5",
    transform=transform,
    end_tp=end_tp,
)

# %%
# idx = 50
# _, y = test_dataset[idx]
# y = y / y.max()
# # show examples of images for each threshold

# fig, axes = plt.subplots(10, 5, figsize=(15, 20))

# for j in range(5):
#     axes[0, j].imshow(y[j].cpu().numpy(), cmap="gray")
#     axes[0, j].axis("off")
# axes[0, 0].set_title("Original")

# for i in range(1, 10):
#     threshold = i / 10
#     binary = y > threshold
#     for j in range(5):
#         axes[i, j].imshow(binary[j].cpu().numpy(), cmap="gray")
#         axes[i, j].axis("off")
#     axes[i, 0].set_title(f"Threshold: {threshold}")


# # %%

# %%
# Pass one input through all four models and show images
import torch
from dlmbl_unet import UNet

unet_depth = 3
in_channels = 5
out_channels = 5
# Create the model
model = UNet(
    depth=unet_depth,
    in_channels=in_channels,
    out_channels=out_channels,
)
model = model.cuda()
model.eval()
# %%
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

n_bins = 100  # Number of bins in the colormap

input_data = test_dataset[30][0].unsqueeze(0).cuda().float()
input_gt = test_dataset[30][1].unsqueeze(0).cuda().float()
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for j in range(5):
    axes[j].imshow(input_data[0, j].cpu().numpy(), cmap="gray")
    axes[j].axis("off")
plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
colormap = LinearSegmentedColormap.from_list(
    compartment, ["black", colors[0]], N=n_bins
)
for j in range(5):
    axes[j].imshow(input_gt[0, j].cpu().numpy(), cmap=colormap)
    axes[j].axis("off")
plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)

fig1, axes1 = plt.subplots(1, 5, figsize=(15, 3))
plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)
for i, (train_date, gfp_type, _, _) in enumerate(experiments):
    experiment_dir = output_dir / f"{train_date}_train_{gfp_type}"
    model_checkpoint = torch.load(experiment_dir / "best_model.pth")
    model.load_state_dict(model_checkpoint["model"])
    with torch.no_grad():
        output = model(input_data)
        # output = torch.sigmoid(output)
        output = output.cpu().numpy()

    # fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    # Create colormap that goes from black to the color of the compartment
    colormap = LinearSegmentedColormap.from_list(
        compartment, ["black", colors[i]], N=n_bins
    )
    vmin = output.min()
    vmax = output.max()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for j in range(5):
        axes1[j].imshow(
            output[0, j],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            alpha=np.clip(output[0, j], 0, 1),
        )
        axes1[j].axis("off")
        axes[j].imshow(output[0, j], cmap=colormap, vmin=vmin, vmax=vmax)
        axes[j].axis("off")
        # axes[0].set_title(gfp_type)
        plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)

# %%
