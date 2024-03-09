import math

import cv2
import os
from enum import Enum
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence


class DatasetType(Enum):
    Train = 'train'
    Test = 'test'
    Validation = 'validation'


class SkinLesionType(Enum):
    Benign = 'Benign'
    Malignant = 'Malignant'


class ColorSpace(Enum):
    BGR = 'BGR'
    HSV = 'HSV'
    LAB = 'LAB'
    YCbCr = 'YCbCr'
    Greyscale = 'Greyscale'


class SkinLesionDataSequence(Sequence):
    def __init__(self, base_dir, dataset_type, color_space, batch_size=32, normalize=True, augment=False, augmentation_factor=3):
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.color_space = color_space
        self.batch_size = batch_size
        self.normalize = normalize
        self.augment = augment
        self.augmentation_factor = augmentation_factor if augment else 1
        self.img_files, self.labels = self._load_image_paths_and_labels()
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips with a 50% chance
            iaa.Flipud(0.5),  # vertical flips with a 50% chance
            iaa.Affine(rotate=(-20, 20)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        ])

    def _load_image_paths_and_labels(self):
        paths = {
            SkinLesionType.Benign: os.path.join(self.base_dir, self.dataset_type.value,
                                                f"{SkinLesionType.Benign.value}Preprocessed",
                                                self.color_space.value),
            SkinLesionType.Malignant: os.path.join(self.base_dir, self.dataset_type.value,
                                                   f"{SkinLesionType.Malignant.value}Preprocessed",
                                                   self.color_space.value)
        }
        img_files = []
        labels = []
        for category, path in paths.items():
            for img_file in os.listdir(path):
                if img_file.endswith('.jpg') or img_file.endswith('.jpeg'):
                    img_files.append(os.path.join(path, img_file))
                    labels.append(0 if category == SkinLesionType.Benign else 1)
        return img_files, labels

    def __len__(self):
        # Adjusted to account for augmentation_factor
        total_images = len(self.img_files) * self.augmentation_factor
        return math.ceil(total_images / self.batch_size)

    def __getitem__(self, idx):
        # Adjust the start and end index to account for the augmented images
        start_idx = (idx * self.batch_size) // self.augmentation_factor
        end_idx = ((
                               idx + 1) * self.batch_size + self.augmentation_factor - 1) // self.augmentation_factor

        batch_files = self.img_files[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # Load and optionally augment images
        batch_images = []
        for file in batch_files:
            image = self.load_image(file)
            if self.augment:
                # Create augmented versions of the image
                augmented_images = [self.augmenter.augment_image(image) for _ in
                                    range(self.augmentation_factor)]
                batch_images.extend(augmented_images)
            else:
                batch_images.append(image)

        # Ensure the batch size is correct, especially for the last batch
        batch_images = batch_images[:self.batch_size]
        batch_labels = np.repeat(batch_labels, self.augmentation_factor)[
                       :self.batch_size]

        if self.normalize:
            batch_images = [self.normalize_image(img) for img in batch_images]

        return np.array(batch_images), np.array(batch_labels)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return image

    def augment_image(self, image):
        image = np.expand_dims(image, 0)
        image = next(self.augmentation.flow(image, batch_size=1))[0]
        return image

    def load_and_process_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Ensure image is resized to 224x224
        if self.augment:
            image = self.augment_image(image)
        if self.normalize:
            image = self.normalize_image(image)
        return image

    def normalize_image(self, image):
        MinI = np.min(image)
        MaxI = np.max(image)
        I_norm = ((image - MinI) * (2 / (MaxI - MinI))) - 1
        return I_norm

