import cv2
import os
from tqdm import tqdm

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)


def convert_color_spaces(image):
    conversions = {
        'HSV': cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        'LAB': cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
        'YCbCr': cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb),
        'Greyscale': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    }
    return conversions


def preprocess_and_convert_image(image_path):
    img = cv2.imread(image_path)
    preprocessed_img = apply_gaussian_blur(img)
    preprocessed_img = apply_clahe(preprocessed_img)
    converted_images = convert_color_spaces(preprocessed_img)
    converted_images['BGR'] = preprocessed_img
    return converted_images


def save_image(image, path):
    """Ensures grayscale images have correct dimensions for saving."""
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, image)


def process_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_files = [f for f in os.listdir(input_directory) if
                   os.path.isfile(os.path.join(input_directory, f))]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_directory, image_file)
        converted_images = preprocess_and_convert_image(input_path)

        for color_space, image in converted_images.items():
            color_space_dir = os.path.join(output_directory, color_space)
            if not os.path.exists(color_space_dir):
                os.makedirs(color_space_dir)
            output_path = os.path.join(color_space_dir, image_file)
            save_image(image, output_path)


if __name__ == "__main__":
    directories = [
        ('test/Malignant', 'test/MalignantPreprocessed'),
        ('test/Benign', 'test/BenignPreprocessed'),
        ('train/Malignant', 'train/MalignantPreprocessed'),
        ('train/Benign', 'train/BenignPreprocessed'),
    ]

    for input_directory, output_directory in directories:
        process_images_in_directory(input_directory, output_directory)
        print(
            f"Processing complete for {input_directory}. Processed images are saved in: {output_directory}")
