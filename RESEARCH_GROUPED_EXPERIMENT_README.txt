BE Plant Disease ViT — Group-Aware Evaluation Bundle

PURPOSE
This bundle preserves your existing baseline and adds a second, more careful experiment. The new experiment uses explicit train/validation/test manifests whose derived leaf-group identifiers do not cross between splits. This lets us test whether the very high image-level-split accuracy also holds when related leaf images are kept together.

IMPORTANT
- Do NOT overwrite or delete your existing train.py, dataset.py, evaluate.py, checkpoint, or results.
- This bundle does NOT contain model weights and does NOT train automatically.
- The custom split is metadata-based. It guarantees separation of the derived group identifiers, but is not an independent visual inspection proving every image is a distinct physical leaf.
- The original 99.59% result must remain labelled as the original random image-level split result.

FILES
- create_group_aware_split.py: reproducible splitter using color_train.txt, color_test.txt and leaf-map.json.
- grouped_split/train.txt, val.txt, test.txt: manifests already generated for your 54,305-image dataset.
- grouped_split/split_report.json: counts, class distribution, and overlap validation.
- grouped_split/split_details.csv: per-image assignment and derived group identifier.
- dataset_grouped.py: explicit-manifest data loader.
- train_grouped.py: separate training entry point; copies training/model hyperparameters from your existing baseline checkpoint to make the comparison fair.
- evaluate_grouped.py: evaluates the grouped-trained checkpoint on the explicit grouped test set. It deliberately does not draw the direct base-paper-vs-ours comparison graph because the experimental split protocols differ.

GENERATED SPLIT
- Training: 38,013 images (~70.00%), 14,015 derived groups
- Validation: 8,144 images (~15.00%), 3,000 derived groups
- Test: 8,148 images (~15.00%), 3,000 derived groups
- All 38 classes are present in all splits.
- Train/validation, train/test, and validation/test group overlap: zero for the derived identifiers.

HOW TO USE
1. Extract the contents of this ZIP into your project root:
   D:\Applications\Codes\plant_disease_vit
   Keep the existing files. The new Python files have different names.

2. Verify these files exist:
   grouped_split\train.txt
   grouped_split\val.txt
   grouped_split\test.txt
   checkpoints\best_model.pt

3. Start the grouped-split training run from PowerShell in the project root:

   python train_grouped.py --data_dir "D:\Applications\Codes\plant_disease_vit\datasets\plantvillage dataset\color" --split_dir ".\grouped_split" --match_checkpoint_config ".\checkpoints\best_model.pt" --output_dir ".\checkpoints_grouped"

   The script reads the baseline checkpoint's saved config and copies its model/training hyperparameters. It initializes a NEW model; it does not start from the baseline weights. Outputs go into checkpoints_grouped, preserving the original checkpoint.

4. After training finishes, evaluate the new best checkpoint:

   python evaluate_grouped.py --checkpoint ".\checkpoints_grouped\best_model.pt" --data_dir "D:\Applications\Codes\plant_disease_vit\datasets\plantvillage dataset\color" --split_dir ".\grouped_split" --output_json ".\checkpoints_grouped\eval_results_grouped.json"

5. Upload checkpoints_grouped\eval_results_grouped.json and checkpoints_grouped\history.json here.

EXPECTED RUNTIME
Your earlier 50-epoch run took about 2.4 hours. This second run may take a similar amount of time. Keep the laptop plugged in and prevent sleep during training. If it is interrupted, do not restart from scratch. Re-run the same command and add `--resume ".\checkpoints_grouped\last.pt"`. Keep the same --split_dir, --match_checkpoint_config, and --output_dir.

REPRODUCIBILITY NOTE
The script creates a 70/15/15 split by applying StratifiedGroupKFold (20 folds, shuffle=True, seed 42) to the union of the uploaded official train/test manifests. It uses the PlantVillage Hugging Face loader's leaf_id assignment rules. Folds 0–2 are validation, 3–5 are test, and 6–19 are training. Existing official split overlaps caused by missing-map fallback identifiers are therefore not carried across our new partitions.
