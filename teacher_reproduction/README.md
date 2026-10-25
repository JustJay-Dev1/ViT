# Teacher Reproduction — PlantVillage Vision Transformer

This folder is a clean, separate training setup. It is intended to live at:

`D:\Applications\Codes\plant_disease_vit\teacher_reproduction`

It does not overwrite the original project's `model.py`, `train.py`, `checkpoints\best_model.pt`, or saved 99.59% baseline result.

## Reference

A. Ouamane et al., “Optimized Vision Transformers for Superior Plant Disease Detection,” *IEEE Access*, 2025. DOI: [10.1109/ACCESS.2025.3547416](https://doi.org/10.1109/ACCESS.2025.3547416).

Dataset candidate: [Project-AgML/plant_village_classification](https://huggingface.co/datasets/Project-AgML/plant_village_classification), listed as 55,448 images across 39 classes, including `Background_without_leaves`.

## Reproduction status and limitations — read before training

This bundle aims to reproduce the teacher architecture and the paper's **reported** configuration as closely as its text supports. It is not possible to reconstruct the authors' exact experiment from the article alone because important artifacts are not published (including exact split manifests and seed), and some reported dataset counts do not reconcile.

The paper's Table I reports PlantVillage counts of 44,379 train + 5,531 validation + 5,574 test = 55,484 images. The public 39-class dataset above lists 55,448 images, a difference of 36. This setup never duplicates images to force the paper's inconsistent total. It creates a seeded, class-stratified 80/10/10 split of all actual source rows; with 55,448 rows this should be 44,358 train, 5,545 validation and 5,545 test. This is a reproducible approximation, **not the exact original split**.

The Hugging Face 39-class source does not expose a `leaf_id`/physical-leaf grouping field. Therefore, the fixed split guarantees no row-index overlap, but it cannot guarantee that repeated photographs of the same physical leaf never cross partitions. The paper does not publish enough split metadata to reconstruct its exact split from the available files. This must be disclosed in any report; do not describe this split as leaf-grouped.

The paper's reported settings used here are: 224x224 input, 16x16 patches, dim 512, depth 6, 8 heads, MLP dim 1024, sinusoidal positional encoding, ReLU in the feed-forward network, post-normalization around the residual sublayers, AdamW, learning rate 1e-4, effective batch size 64, and 1000 epochs.

The paper does not fully specify dropout probability, all affine/crop magnitudes, weight decay, a learning-rate scheduler, initialization, and all details around how augmentation was counted. This bundle makes those gaps explicit: dropout is set to 0.0; AdamW uses PyTorch's default weight decay 0.01; learning rate is fixed because no scheduler is stated in the paper; affine/crop magnitudes and ImageNet normalization are implementation choices. Augmentation is applied online to training examples only, not to validation/test. This is more suitable for an uncontaminated evaluation than augmenting or multiplying held-out test images.

The paper refers to its PlantVillage protocol from another work but gives no image manifest, split seed, or downloadable code. Therefore, do not claim exact data-split reproduction. In addition, the paper's table counts and image-count prose are internally inconsistent; report our actual verified counts.

## How to use this folder

1. Extract the bundle into your local `plant_disease_vit` root as `teacher_reproduction`.
2. Upload the **extracted** `teacher_reproduction` folder to Google Drive at `MyDrive/plant_disease_vit/teacher_reproduction` (keep the `src`, `configs`, `splits`, and `colab` folders).
3. Open `colab/teacher_reproduction_colab.ipynb` in Google Colab.
4. Run the setup and preflight cells first. They download/inspect the dataset, create a fixed split manifest in Drive, print the class and split counts, and benchmark one forward/backward step. They do not start the 1000-epoch run automatically.
5. Review the printed dataset/split report. Only then run the training cell and type `TRAIN` when prompted.
6. Checkpoints are saved to Drive. Every 5 epochs the latest resumable checkpoint is written; the best validation checkpoint is separately saved when validation accuracy improves. After reconnecting, rerun setup and the training cell to resume from `last.pt`.
7. After training completes, run the evaluation cell. The test split must not be used to select epochs or tune settings.

## Default Colab training settings

- GPU required; training deliberately aborts rather than running accidentally on CPU.
- Micro-batch 16, accumulation steps 4 = effective batch 64.
- FP16 mixed precision.
- AdamW with LR 1e-4.
- 1000 epochs, no early stopping, no scheduler (no scheduler is specified in the paper).
- Checkpoint every 5 epochs to reduce slow Drive writes; `best_model.pt` saves on every new best validation result.

If the preflight benchmark reports CUDA out-of-memory, lower micro-batch to 8 and change accumulation to 8 (effective batch remains 64). Do not change image size/model dimensions just to make it fit unless we record the change as a hardware adaptation.

## Outputs

- `splits/plant_village_39class_seed42.json`: exact row indices for each split and source fingerprint.
- `checkpoints/last.pt`: resumable state (model, optimizer, scaler, epoch, history and RNG state).
- `checkpoints/best_model.pt`: best validation checkpoint, includes class names and model config for later knowledge distillation.
- `results/history.json`: training and validation history, updated after every epoch.
- `results/test_metrics.json`: test metrics and per-class results after final evaluation.

## Important for knowledge distillation

Use the saved `best_model.pt` as the teacher. It contains raw model weights and the exact 39-class label order. The later student must use the same class order. Keep the teacher checkpoint unchanged after training; do not replace it with student weights.
