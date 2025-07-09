# MABY-Thesis: Yeast Compartment Segmentation with U-Net and Classical Methods

This repository contains the codebase used for my bachelor thesis project:  
**"Deep Learning vs Classical Segmentation: A Study on Yeast Cell Compartments"**

It extends [adjavon/maby](https://github.com/adjavon/maby) with a full training and evaluation pipeline for U-Net segmentation and comparisons with classical methods like Cellpose and Otsu thresholding.

🔗 Repository: [https://github.com/lknoche/maby-thesis](https://github.com/lknoche/maby-thesis)

---

## Project Structure:
maby-thesis/
├── README.md
├── requirements.txt
├── train.py                      # Train U-Net on tile-based images
├── test.py                       # Evaluate trained model with F1 and similarity
├── dataset.py                    # Dataset loading and augmentation
├── image.py                      # Image handling and preprocessing
├── download_and_run.py           # Downloads from OMERO and runs training
├── alibylite.org                 # Optional config for OMERO/local paths

├── export/
│   ├── export_fluorescence_tiles.py
│   ├── export_ground_truth_masks.py
│   ├── export_gt_fluorescence_images.py
│   ├── export_unet_tile_masks.py

├── extract/
│   ├── extract_bf_unetlike.py
│   ├── extract_fluo.py
│   ├── extract_selected_fluorescence_images.py
│   ├── extract_unet_like_tiles.py
│   ├── extract_unet_like_tiles_vacuole.py
│   ├── extract_z_slices_for_qpi.py

├── overlays/
│   ├── make_combined_overlay_grid_vacuole.py
│   ├── overlay_cellpose_vph1.py
│   ├── run_cellpose_vph1.py
│   ├── general_segmentation_otsu.py

├── plots/
│   └── plot_results.py



##How to use:
1.Set up environment:
git clone https://github.com/lknoche/maby-thesis.git
cd maby-thesis
pip install -r requirements.txt
2. Preprocess Data:
If working with OMERO:
Use download_and_run.py to download and train:
python download_and_run.py -g Htb2_sfGFP
otherwise run train.py directly with the right directory:
python train.py -b /path/to/data -g Vph1_GFP
3. Evaluation:
python test.py  -g Vph1_GFP -m model_19.pt
4. Visualize Results:
python plots/plot_results.py

