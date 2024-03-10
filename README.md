# Skin Lesion Cancer Classification

## Introduction

The task at hand was to develop a computer vision model for classifying skin lesions as either benign or malignant (cancerous). This is a binary classification problem with potential applications in the medical field, assisting in the early detection of skin cancer.

The most important and final notebook here is `final_data_augmentation.ipynb`. It contains the final implementation, which incorporates data augmentation techniques. The notebook loads the previously trained models, evaluates their performance on the test set, and calculates various metrics (accuracy, precision, recall, and F1-score) for each individual model and ensemble approaches.

Please read `HowTo.md` for a detailed explanation on how to run the code.

## Dataset

The dataset consisted of images of skin lesions, divided into two categories: benign and malignant. The images were further split into train, test, and validation sets. The preprocessing steps involved converting the images to different color spaces (BGR, HSV, LAB, YCbCr, and Grayscale) and applying techniques like Gaussian blurring and Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the image quality.

## Preprocessing

The preprocessing steps were carried out in several Python scripts:

- `preprocess.py`: This script applied Gaussian blurring and CLAHE to the images in the train and test directories, storing the preprocessed images in separate directories.
- `color_conversions.py`: This script converted the preprocessed images to different color spaces (BGR, HSV, LAB, YCbCr, and Grayscale) and saved them in respective directories.
- `create_validation_set.py`: This script created a validation set by moving a subset of images from the test set to a new validation directory.
## Data Generator
The `data_generator.py` script implemented a custom data generator (SkinLesionDataSequence) to load and preprocess the images during training. It handled tasks such as shuffling, normalization, and data augmentation (including horizontal and vertical flips, rotations, and Gaussian noise). The data generator was designed to work with different color spaces and dataset types (train, test, and validation).

## Model Training
The `data_generator.py` script also contained code for training and evaluating the models. A convolutional neural network (CNN) architecture was used, with layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout. The models were trained separately for each color space, with the best model for each color space being saved.

## Final Data Augmentation
The `final_data_augmentation.ipynb` notebook contains the final implementation, which incorporates data augmentation techniques. The notebook loads the previously trained models, evaluates their performance on the test set, and calculates various metrics (accuracy, precision, recall, and F1-score) for each individual model and ensemble approaches.

Two ensemble methods were explored:
- **Simple Average Ensemble:** The predictions from individual models were averaged to obtain the ensemble prediction.
- **Bayesian Model Averaging (BMA) Ensemble:** The predictions from individual models were weighted based on their accuracy scores and averaged to obtain the ensemble prediction.
Both ensemble methods achieved similar results, with an accuracy of around 94% on the test set.

## Results
The notebook final_data_augmentation.py presents the final results, including a table summarizing the performance metrics for individual models and ensemble approaches. The best overall score was around 94% accuracy, achieved by both the Simple Average Ensemble and the Bayesian Model Averaging Ensemble.
Additionally, the notebook mentions that attempts were made to remove hair from the images, but this step worsened the overall score.

| Model                       | Accuracy | Precision | Recall | F1       |
|-----------------------------|----------|-----------|--------|----------|
| BGR                         | 0.926875 | 0.935032  | 0.9175 | 0.926183 |
| HSV                         | 0.920625 | 0.925411  | 0.9150 | 0.920176 |
| LAB                         | 0.923750 | 0.942559  | 0.9025 | 0.922095 |
| YCbCr                       | 0.920625 | 0.904934  | 0.9400 | 0.922134 |
| Greyscale                   | 0.920625 | 0.925411  | 0.9150 | 0.920176 |
| Ensemble                    | 0.938125 | 0.938673  | 0.9375 | 0.938086 |
| Bayesian Model Averaging    | 0.938125 | 0.938673  | 0.9375 | 0.938086 |

One interesting observation is, that the precision of LAB model is the highest, but the recall is the lowest. This means that the LAB model is the best at correctly identifying malignant lesions, but it also has the highest number of false negatives (benign lesions classified as malignant). This is an important consideration, as it could have serious implications in a real-world medical setting.

In a real world scenario that could be the wanted behavior, as it is better to have a false positive than a false negative. In the case of skin cancer, it is better to have a benign lesion classified as malignant, as it would lead to further examination and not to a missed diagnosis.

## Possible future steps
As all images have a fairly low quality (224x224x3), I experimented with using GANs to improve the quality of the images.
In particular [ESRGAN](https://github.com/xinntao/Real-ESRGAN) seems very promising and was alsop successful on individual tries. I didn't use it in the final model, for time reasons and simple
space constraints on my machine. But using it would lead to further possibilities, for example for images where the skin lesion is one a small part of the total image, to crop the image and then use ESRGAN to improve the quality of the lesion.