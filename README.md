Participating in this Kaggle competition was an important experience for me. I initially ranked 16th out of 2700 participants on the public leaderboard, which was exciting. However, when the private leaderboard was revealed, I dropped to 215th place. This was likely due to poor submission selection and perhaps too much hyperparameter tuning because my approach was completely leak-free. (My other submission which I unselected just 3 minutes before the deadline could have won a silver medal.)


# Skin Cancer Detection Using SLICE-3D Dataset

This repository contains the code and model implementation for the SLICE-3D dataset from the Kaggle competition. The task involves predicting the probability that a skin lesion is malignant, based on diagnostically labeled images and associated metadata. The model leverages both image and tabular data, including Out-of-Fold (OOF) predictions for stacking.


## Overview

In this solution, we use a combination of image-based models and tabular data to predict the probability that a lesion is malignant. The image data consists of 3D Total Body Photography (TBP) cropped lesion images. To enhance prediction accuracy, we generate OOF predictions from an EfficientNet-based image model and combine them with tabular data for a final ensemble using Gradient Boosted Decision Trees (GBDT).

## Configuration

The configuration settings for the training process are defined in the `CONFIG` dictionary:

```python
CONFIG = {
    "seed": 42,
    "epochs": 15,
    "img_size": 336,
    "model_name": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "train_batch_size": 64,
    "valid_batch_size": 64,
    "learning_rate": 1e-4,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 14,
    "T_0": 14,
    "weight_decay": 1e-6,
    "fold": 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

## Handling Class Imbalance

Given the extreme class imbalance, we perform stratified sampling to maintain the ratio of positive to negative cases:

```python
df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)

## Cross-Validation Strategy

We use Stratified Group K-Fold cross-validation to ensure that:

The target distribution is balanced across folds.
Patients (groups) are not split between training and validation sets to prevent data leakage.

```python
sgfk = StratifiedGroupKFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])
df["kfold"] = -1

for fold, (train_idx, val_idx) in enumerate(sgfk.split(df, df.target, df.patient_id)):
    df.loc[val_idx, "kfold"] = int(fold)


