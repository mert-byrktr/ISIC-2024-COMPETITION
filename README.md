Participating in this Kaggle competition was an important experience for me. I initially ranked 16th out of 2700 participants on the public leaderboard, which was exciting. However, when the private leaderboard was revealed, I dropped to 215th place. This was likely due to poor submission selection and perhaps too much hyperparameter tuning because my approach was completely leak-free. (My other submission which I unselected just 3 minutes before the deadline could have been in the 30th place.)


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
```

## Handling Class Imbalance

Given the extreme class imbalance, we perform stratified sampling to maintain the ratio of positive to negative cases:

```python
df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)
```

## Cross-Validation Strategy

We use Stratified Group K-Fold cross-validation to ensure that:

* The target distribution is balanced across folds.
* Patients (groups) are not split between training and validation sets to prevent data leakage.

Same cv strategy is applied to both OOF stacking predictions and tabular models.

```python
sgfk = StratifiedGroupKFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])
df["kfold"] = -1

for fold, (train_idx, val_idx) in enumerate(sgfk.split(df, df.target, df.patient_id)):
    df.loc[val_idx, "kfold"] = int(fold)
```
## Augmentations
We applied extensive augmentation techniques from previous winning solutions of the ISIC 2020 competition using the albumentations library to enrich the training data:

```python
data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.CoarseDropout(max_height=int(CONFIG['img_size']*0.375),
                        max_width=int(CONFIG['img_size']*0.375), num_holes=1, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=0,
                           p=0.85),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=0.75),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2()], p=1.),
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2()], p=1.)
}

```

## Model Architecture
EfficientNet with GeM Pooling
We used the EfficientNet architecture with a Generalized Mean (GeM) pooling layer for better feature aggregation.
The code is flexible to accommodate other architectures such as EVA02, ResNet, ResNeXt, etc.

## Loss Function

We used Binary Cross-Entropy Loss for this binary classification task. We also tried FocalLoss, but the target mean was much higher than expected.

## Optimizer and Scheduler
We used the Adam optimizer with a learning rate scheduler combining gradual warmup and cosine annealing

## Tabular Model with OOF Stacking
After generating OOF predictions from the image model, we use these predictions as additional features in a tabular model that includes Gradient Boosted Decision Trees (GBDT). The model is an ensemble of LightGBM, CatBoost, and XGBoost models, with OOF stacking predictions integrated into the tabular features.

## Cross-Validation and Fold Distribution

We used **Stratified Group K-Fold Cross-Validation** to ensure that:

- The target distribution (malignant vs. benign) is balanced across folds.
- No patient appears in both training and validation sets, thereby preventing data leakage.

To verify the distribution of target labels and unique patients across the folds, we calculated the mean target ratio, the count of samples, and the number of unique patients in each fold:

```python
# Verify the distribution across folds
print(df_train.groupby('fold').agg({
    'target': ['mean', 'count'],
    'patient_id': 'nunique'
}))
```

## Feature Engineering
#### 1. **Patient-Level Normalization**

Given the dataset includes multiple lesions per patient, we introduced several **patient-level normalization** features to account for variations in lesion characteristics across patients:

- **Patient-Level Mean Normalization**: For each lesion feature, we computed the mean for each patient and normalized the feature by subtracting the mean and dividing by the standard deviation for that patient.
  - Example: `lesion_size_ratio_patient_norm = (lesion_size_ratio - mean_lesion_size_ratio_per_patient) / std_lesion_size_ratio_per_patient`
- **Patient-Level Sum Ratio**: For each feature, we calculated the ratio of a lesion's value to the sum of all lesions for that patient.
- **Patient-Level Min-Max Scaling**: Features were scaled within each patient using the min-max range of each feature.
- **Patient-Level Ordinal Ranking**: Each lesion's feature was ranked within the patient cohort, helping to capture relative ordering of features like lesion size and color variation.
- **Patient-Level Quantile Scaling**: Each lesion feature was quantile-scaled within the patient's lesion set, ensuring consistent distribution of values across patients.

#### 2. **Ugly Duckling Processing**

To identify outliers (anomalous lesions) within a patient, we applied **Ugly Duckling processing**:

- **Z-Score Calculation**: We calculated z-scores for all lesion features within a patient's group, comparing each lesion to others from the same patient.
  - Example: `ud_size_ratio = |z-score(lesion_size_ratio)|`
- **Location-Specific Ugly Ducklings**: We extended this process by grouping lesions by their anatomical site and calculating z-scores for each feature within those subgroups.
- **Percentile-Based Ugly Duckling Scores**: We also ranked lesions within the patient by percentiles, capturing how extreme a lesion's characteristics were relative to others.
- **Ugly Duckling Count**: We counted how many features of a lesion exceed a certain threshold (e.g., z-score > 2), flagging highly anomalous lesions.
- **Severity and Consistency**: Features like "ugly duckling severity" and "ugly duckling consistency" captured how extreme and how consistent these anomalies were across the lesion set.

  
#### 3. **Handling Highly Correlated Features**

To prevent multicollinearity and improve model performance, we dropped features that were highly correlated with one another. This step was crucial because many of the engineered features, particularly those related to lesion size, color, and shape, exhibited strong correlations.

We calculated the correlation matrix of all numerical features and applied a threshold of 0.91 to identify and remove highly correlated features:

```python
def select_features_using_corr_matrix(df, threshold=0.91):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = df.columns.difference(to_drop)
    return selected_features.tolist()
```

#### 4. **Out-of-Fold (OOF) Stacking Features**

To further improve the tabular model's performance, we integrated OOF predictions from the image models as additional features in the tabular data. These predictions were generated using models like EfficientNet and EVA02, and were subsequently used for feature stacking in the Gradient Boosted Decision Tree (GBDT) models.

```python
df_effb0_oof = pd.read_csv(oof_path)
df_effb0_oof = df_effb0_oof[['oof_predictions_effnetb0']].reset_index(drop=True)
df = df.reset_index(drop=True)
df['oof_predictions_effnetb0'] = df_effb0_oof['oof_predictions_effnetb0']
feature_cols.append('oof_predictions_effnetb0')

```
##. **Methods That Didn't Work**

Throughout the experimentation process, several approaches were tested but did not lead to meaningful improvements in model performance. These include the following:

#### 1. **Image Models**
- **Focal Loss**: Focal Loss was tested as an alternative to Binary Cross-Entropy (BCE) to address the extreme class imbalance. However, it did not improve model performance compared to BCE. The highly imbalanced dataset likely required different handling techniques, and Focal Loss introduced instability during training.
  
- **ResNet Architectures for OOF Stacking**: We experimented with ResNet architectures for generating OOF predictions to be used in the tabular model. These architectures performed worse than the EfficientNet and EVA02 models, and their OOF predictions did not enhance the tabular model.

#### 2. **Feature Selection**
- **SHAP and LOFO (Leave-One-Feature-Out)**: Feature selection using SHAP values and LOFO was explored to identify the most important features. While theoretically sound, these methods did not lead to a meaningful improvement in model performance. They failed to outperform basic correlation-based feature selection due to the complexity of interactions between features.

#### 3. **Optimizers**
- **Trying Different Optimizers**: Beyond Adam, we experimented with other optimizers such as SGD and RMSprop. However, these alternative optimizers did not perform as well as Adam in terms of convergence speed and model performance on validation data.

#### 4. **Learning Rate Schedulers**
- **GradualWarmupScheduler**: While combining gradual warmup with cosine annealing was tested, it did not significantly outperform the default CosineAnnealingLR scheduler on validation performance. 
- **CosineAnnealingWarmRestarts**: This was also tested to handle cyclic learning rates, but it did not yield improvements in model generalization.

#### 5. **Augmentations**
- **Hair Removal**: Augmentation techniques that removed hair from skin lesion images were attempted, but this approach did not significantly improve the image models' ability to distinguish between benign and malignant lesions.
- **Microscope Augmentation**: Simulating microscope effects as an augmentation technique was also tested, but it failed to enhance the model's understanding of lesion characteristics and did not improve accuracy.

#### 6. **Data Sampling**
- **Downsampling Negative Images Other Than 1/50 Ratio**: Different downsampling ratios for negative samples were tested to reduce class imbalance (e.g., 1/20 and 1/10), but the 1/50 ratio worked best. These other ratios resulted in either underfitting or poor generalization.

#### 7. **Categorical Feature Encoding**
- **Ordinal Encoder for Categorical Features**: We experimented with using ordinal encoding for categorical features like `anatom_site_general` and `sex`, but it performed worse than One-Hot Encoding. This is likely because ordinal relationships between categories were non-existent or irrelevant in the dataset.

#### 8. **Adding Previous Competition Images**
- **Using Previous Competition Images**: We attempted to incorporate images from previous ISIC competitions to increase the dataset size. However, this approach did not work well because the images from this year’s competition were of much lower resolution, whereas the older images were high-resolution dermoscopic images. This discrepancy in resolution led to inconsistencies in model training and hurt overall performance.

