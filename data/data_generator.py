import math

import cv2
import os
from enum import Enum
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle


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
    def __init__(self, base_dir, dataset_type, color_space, batch_size=32,
                 normalize=True, augment=False, augmentation_factor=3):
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
            SkinLesionType.Malignant: os.path.join(self.base_dir,
                                                   self.dataset_type.value,
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
        img_files, labels = shuffle(img_files, labels, random_state=42)
        return img_files, labels

    def __len__(self):
        # Adjusted to account for augmentation_factor
        total_images = len(self.img_files) * self.augmentation_factor
        return math.ceil(total_images / self.batch_size)

    def __getitem__(self, idx):
        start_idx = (idx * self.batch_size) // self.augmentation_factor
        end_idx = ((
                               idx + 1) * self.batch_size + self.augmentation_factor - 1) // self.augmentation_factor

        batch_files = self.img_files[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        batch_images = []
        batch_labels_new = []
        for file, label in zip(batch_files, batch_labels):
            image = self.load_image(file)
            if self.augment:
                # Create augmented versions of the image
                new_images = [image]
                augmented_images = [self.augmenter.augment_image(image) for _ in
                                    range(self.augmentation_factor - 1)]
                new_images.extend(augmented_images)
                batch_images.extend(new_images)
                batch_labels_new.extend([label] * self.augmentation_factor)
            else:
                batch_images.append(image)
                batch_labels_new.append(label)

        # Ensure the batch size is correct, especially for the last batch
        batch_images = batch_images[:self.batch_size]
        batch_labels_new = batch_labels_new[:self.batch_size]

        if self.normalize:
            batch_images = [self.normalize_image(img) for img in batch_images]

        return np.array(batch_images), np.array(batch_labels_new)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return image

    def normalize_image(self, image):
        minI = np.min(image)
        maxI = np.max(image)
        i_norm = ((image - minI) * (2 / (maxI - minI))) - 1
        return i_norm
