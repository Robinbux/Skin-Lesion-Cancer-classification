import os
import shutil


def create_validation_set(base_dir='data', split_ratio=0.5):
    test_dir = os.path.join(base_dir, 'test')
    validation_dir = os.path.join(base_dir, 'validation')

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    categories = ['Benign', 'Malignant']
    color_spaces = ['BGR', 'HSV', 'LAB', 'YCbCr', 'Greyscale']

    for category in categories:
        for color_space in color_spaces:
            src_dir = os.path.join(test_dir, f'{category}Preprocessed', color_space)
            dest_dir = os.path.join(validation_dir, f'{category}Preprocessed',
                                    color_space)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            images = [f for f in os.listdir(src_dir) if
                      f.endswith('.jpg') or f.endswith('.jpeg')]

            num_validation_images = int(len(images) * split_ratio)

            for img in images[:num_validation_images]:
                shutil.move(os.path.join(src_dir, img), os.path.join(dest_dir, img))


if __name__ == "__main__":
    create_validation_set(base_dir='', split_ratio=0.2)
