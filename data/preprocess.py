import cv2
import os
from tqdm import tqdm

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = apply_gaussian_blur(img)
    img = apply_clahe(img)
    return img


def save_image(image, path):
    cv2.imwrite(path, image)


def process_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_files = [f for f in os.listdir(input_directory) if
                   os.path.isfile(os.path.join(input_directory, f))]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, image_file)

        preprocessed_img = preprocess_image(input_path)
        save_image(preprocessed_img, output_path)


if __name__ == "__main__":
    directories = [
        ('test/Malignant', 'test/MalignantPreprocessed'),
        ('test/Benign', 'test/BenignPreprocessed'),
        ('train/Malignant', 'train/MalignantPreprocessed'),
        ('train/Benign', 'train/BenignPreprocessed'),
    ]

    for input_directory, output_directory in directories:
        process_images_in_directory(input_directory, output_directory)
        print("Processing complete for {}. Processed images are saved in: {}".format(input_directory, output_directory))
