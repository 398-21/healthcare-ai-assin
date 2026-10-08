> **Historical snapshot - superseded (2026-10-08).** This dated working report contains results from the earlier 11,833-stay Task 2 cohort. The current analysis retains all 12,000 stays. Use the repository README and current notebook/DATA_README for active results; the historical numbers below are not current.

---

# GPU Training Instructions for Stage 3 (L4 GBDT)

## Setup

```bash
# Navigate to experiment directory
cd "/mnt/d/personal_Profiles/files/NTU学系资料/Trimester1/Machine Learning for Healthcare AI/Group_project/experiment"

# Create conda environment (only once)
conda env create -f environment_gpu.yml

# Activate environment
conda activate ml_healthcare_gpu

# Verify GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Training Sequence

Run these scripts **in order**:

```bash
# 1. XGBoost training with GPU (20-30 min estimated)
python baselines/run_L4_xgboost_gpu.py

# 2. LightGBM training with GPU (15-25 min estimated)
python baselines/run_L4_lightgbm_gpu.py

# 3. Model comparison and selection
python baselines/run_L4_model_selection.py

# 4. Test evaluation on selected model
python baselines/run_L4_test_evaluation_gpu.py

# 5. SHAP feature importance analysis
python baselines/run_L4_shap_analysis_gpu.py
```

## Expected Outputs

After running all scripts, you should have:

```
results/
├── L4_xgboost_training.json          # XGBoost hyperparameters and validation metrics
├── L4_lightgbm_training.json         # LightGBM hyperparameters and validation metrics  
├── L4_model_selection.json           # Selected model and rationale
├── L4_test_evaluation.json           # Test metrics with bootstrap CI
├── L4_feature_importance.json        # Top features by SHAP
└── figures/
    └── L4_shap_summary.png           # SHAP summary plot
```

Also saved models in:
```
models/
├── L4_xgboost_best.json              # Best XGBoost model
└── L4_lightgbm_best.txt              # Best LightGBM model
```

## Handoff Protocol

When training completes, notify me with:

1. **Completion status**: "All 5 scripts completed successfully" or which one failed
2. **Key metrics**: Report test AUROC and AUPRC from `L4_test_evaluation.json`
3. **Selected model**: XGBoost or LightGBM from `L4_model_selection.json`

Example notification:
```
Training complete. Selected: XGBoost. Test AUROC: 0.8234 [0.7891, 0.8567], AUPRC: 0.5123 [0.4567, 0.5789]
```

Then I will:
- Verify results and check for issues
- Create stage3 process document
- Perform clinical interpretation of SHAP results
- Write self-critique assessment

## Troubleshooting

**If GPU not detected**:
```bash
# Check CUDA installation
nvidia-smi

# Check XGBoost GPU support
python -c "import xgboost as xgb; print(xgb.get_config())"
```

**If out of GPU memory**:
- Scripts include `max_bin=63` to reduce memory
- Can further reduce `n_estimators` or `max_depth` in hyperparameter grid

**If scripts hang**:
- Check `baselines/*.log` files for error messages
- Can reduce cross-validation folds from 5 to 3 in scripts
