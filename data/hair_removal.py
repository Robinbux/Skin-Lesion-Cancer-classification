"""Removing the hair this way somewhat worked, but it made the image quality too bad
and lowered the performance this way. So it's not being used in the final model."""

import cv2
import os
from tqdm import tqdm

def remove_hair_from_image(image_path):
    src = cv2.imread(image_path)
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    _, thresh2 = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)

    return dst


def process_images_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            processed_image = remove_hair_from_image(input_path)
            cv2.imwrite(output_path, processed_image)


if __name__ == "__main__":
    # Test Benign
    # input_directory = 'test/Benign'
    # output_directory = 'test/BenignHairRemoved'
    # Test Malignant
    input_directory = 'test/Malignant'
    output_directory = 'test/MalignantBenignHairRemoved'
    # # Train Benign
    # input_directory = 'train/Benign'
    # output_directory = 'train/BenignHairRemoved'
    # # Train Malignant
    # input_directory = 'train/Malignant'
    # output_directory = 'train/MalignantBenignHairRemoved'

    process_images_in_directory(input_directory, output_directory)
    print("Processing complete. Processed images are saved in:", output_directory)
