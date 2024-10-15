Participating in this Kaggle competition was an important experience for me. I initially ranked 16th out of 2700 participants on the public leaderboard, which was exciting. However, when the private leaderboard was revealed, I dropped to 215th place. This was likely due to poor submission selection and perhaps too much hyperparameter tuning because my approach was completely leak-free. (My other submission which I unselected just 3 minutes before the deadline could have won a silver medal.)


**Skin Cancer Detection Using SLICE-3D Dataset**

This repository contains the code and model implementation for the SLICE-3D dataset from the Kaggle competition. The task involves predicting the probability that a skin lesion is malignant, based on diagnostically labeled images and associated metadata. The model leverages image and tabular data, including Out-of-Fold (OOF) predictions for stacking.

**Overview**

In this solution, we use a combination of image-based models and tabular data to predict the probability that a lesion is malignant. The image data consists of 3D Total Body Photography (TBP) cropped lesion images. To enhance the prediction accuracy, we generate OOF predictions from an EfficientNet-based image model and combine them with tabular data for a final ensemble.


**Augmentations**

We applied the following augmentations to preprocess the image data before feeding it into the model:

* Resize: All images are resized to the configured size (CONFIG['img_size']).
* Normalize: We normalize the pixel values using standard ImageNet mean and standard deviation.

**Model Architecture**

The image model is based on EfficientNetB0 architecture with Generalized Mean (GeM) pooling for feature aggregation. We replaced the default pooling layer with a GeM layer and used a linear layer with sigmoid activation to output the probability of malignancy for each image.

**Out-of-Fold Predictions**

We utilized StratifiedGroupKFold cross-validation with 5 splits, ensuring that the target distribution remains balanced across folds and that patient data is grouped together. OOF predictions were generated for each fold and saved for stacking with tabular data in later steps.
