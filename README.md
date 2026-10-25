# Baseline ViT Reproduction — Plant Disease Detection

Reproduces the optimal Vision Transformer from Ouamane et al., *"Optimized Vision Transformers
for Superior Plant Disease Detection,"* IEEE Access, vol. 13, 2025:
224x224 image, patch 16, dim 512, depth 6, 8 heads, MLP 1024 -> ~13.02M parameters.
This is the **teacher** model that the lightweight student will later be distilled from.

Tested pipeline: training, resume, evaluation and all figures were run end-to-end on synthetic data.
Model verified at 13,031,462 parameters (paper: 13.02M).

## Files

| File | Purpose |
|---|---|
| `model.py` | ViT architecture (with attention capture for heatmaps) |
| `check_model.py` | 5-second sanity check of the architecture |
| `dataset.py` | PlantVillage loading, augmentation, train/val/test split |
| `train.py` | Training: progress bars, AMP, warmup+cosine LR, checkpoints, **resume**, live curves |
| `evaluate.py` | Test-set metrics + all presentation figures |
| `visualize.py` | The plotting code (used by the two scripts above) |

## 1. Dataset — yes, you need it on your PC

Training reads the images from disk. Download once (~1 GB):

- **Kaggle (recommended):** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  (54,305 images, 38 classes). Click Download (free account needed). Unzip it. Inside is
  `plantvillage dataset/` with three folders: `color`, `grayscale`, `segmented`.
  **Use the `color` folder** — it directly contains the 38 class subfolders.
- **No-account alternative:** `git clone https://github.com/spMohanty/PlantVillage-Dataset` and use
  `raw/color`.

Do NOT use the `emmarex/plantdisease` Kaggle dataset: it is a smaller subset with only 15 classes.

Quick check: the path you pass as `--data_dir` should contain folders like `Apple___Apple_scab`,
`Tomato___healthy`, ... (38 of them) and no other nesting.

## 2. Install (Windows, RTX 3050)

```bash
# 1) CUDA build of PyTorch - get the exact command from https://pytorch.org/get-started/locally/
#    (Stable / Windows / Pip / Python / CUDA 12.x). Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) the rest
pip install scikit-learn numpy tqdm pillow matplotlib

# 3) confirm the GPU is visible - must print True
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Run

```bash
python check_model.py                                   # ~13.03M params

# quick pipeline test (a few minutes)
python train.py --data_dir "D:/datasets/plantvillage dataset/color" --epochs 1 --limit_samples 300

# full training
python train.py --data_dir "D:/datasets/plantvillage dataset/color" --epochs 50 --batch_size 32 --amp

# after training: metrics + figures
python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir "D:/datasets/plantvillage dataset/color"
```

- Out of GPU memory? Use `--batch_size 16`.
- GPU sitting idle? Raise `--num_workers` (try 6 or 8); if Windows errors on workers, use 0-2.
- Interrupted? Re-run the same command with `--resume checkpoints/last.pt`.
- Keep the PC awake (disable sleep) and the laptop plugged in and cool during training.
- After epoch 1 the console shows time/epoch, an ETA, and peak GPU memory. Use it to decide the epoch count.
  The learning-rate schedule depends on `--epochs`, so choose it up front rather than stopping early.

## 4. Figures for your presentation

Saved as 200-dpi PNGs in `checkpoints/figures/`:

| Figure | Made by | Slide idea |
|---|---|---|
| `class_distribution.png` | train.py (at start) | The dataset and its class imbalance |
| `training_curves.png` | train.py (refreshed every epoch) | Show it growing live; best epoch starred |
| `paper_comparison.png` | evaluate.py | Our reproduction vs the base paper's numbers |
| `per_class_f1.png` | evaluate.py | Which diseases are hardest |
| `confusion_matrix.png` | evaluate.py | Which diseases get confused with which |
| `correct_predictions.png` | evaluate.py | Sample predictions with confidence |
| `misclassified_examples.png` | evaluate.py | Honest failure cases |
| `attention_maps.png` | evaluate.py | Where the model looks (attention rollout) |

`eval_results.json` also holds accuracy, weighted precision/recall/F1, macro-F1, parameters, size in MB and
batch-1 latency, which are the same axes you will compare the lightweight model on later.

## 5. Report honestly

- The paper does not fully specify its optimizer, LR schedule, epochs or split. This code uses AdamW,
  warmup + cosine, 50 epochs and a random 70/15/15 split, so expect a small gap to 99.77%.
- PlantVillage has ~7 photos of each physical leaf, so a random per-image split lets near-duplicates
  appear in both train and test, which can inflate accuracy. Say so if asked; it supports your
  "benchmark inflation" research gap.
- Attention rollout shows where the model attends. It is a qualitative picture, not proof of what the
  model "understands" and it may highlight the leaf outline rather than the lesion. Scoring
  explanations against annotated lesions is your later explainability work.
