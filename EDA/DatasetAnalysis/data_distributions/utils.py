'''
This module serves data distributions in number of classes and relative object sizes.

Date: 25/10/2023
Author: Alfonso Ponce Navarro
'''
from pathlib import Path
import logging
from .aux_utils import get_pascal_class_distribution, get_pascal_size_distribution
from PIL import Image
IMAGE_EXTENSION_LIST = ['.png', '.jpg', '.jpeg']
LABEL_EXTENSION_LIST = ['.xml', '.json', '.txt']

def number_of_images(images_dir: Path) -> int:
    '''
    Computes the number of images in a data folder. Images must be in images/ folder below task/ folder.

    :param images_dir: Directory of the images
    :return: Number of images in data
    '''
    assert images_dir.exists(), logging.error(f"{images_dir} not found")

    all_images = []
    for extension in IMAGE_EXTENSION_LIST:
        all_images.extend(images_dir.glob(f'*{extension}'))

    return len(all_images)

def number_of_labels(labels_dir: Path) -> int:
    '''
    Computes the number of labels in a data folder. Images must be in labels/ folder below task/ folder.

    :param root_dir: Directory of the labels
    :return: Number of images in data
    '''
    assert labels_dir.exists(), logging.error(f"{labels_dir} not found")

    all_labels = []
    for extension in LABEL_EXTENSION_LIST:
        all_labels.extend(labels_dir.glob(f'*{extension}'))

    return len(all_labels)

def class_distribution(labels_dir: Path) -> dict:
    '''
    Computes class distribution among labels

    :param labels_dir:  directory of the data
    :return: First element is the distribution dictionary. Second element is a list with all class appearances.
    '''
    assert labels_dir.exists(), logging.error(f"{labels_dir} not found")

    for extension in LABEL_EXTENSION_LIST:
        if extension == '.xml':
            distribution_dict, class_ocurrence_list = get_pascal_class_distribution(labels_dir)

    return distribution_dict, class_ocurrence_list

def relative_object_size_distribution(labels_dir: Path) -> tuple:
    '''
    Computes the distribution of sizes and the distribution of images containing different sizes.

    :param labels_dir: Directory of the labels.
    :return: First and second elements refer the object size distribution. Third and
                     fourth element refer to the number of images with different object sizes
    '''
    assert labels_dir.exists(), logging.error(f"{labels_dir} not found")

    for extension in LABEL_EXTENSION_LIST:
        if extension == '.xml':
            object_size_distribution, object_size_occurrences, \
            size_objects_per_image_distribution, size_objects_per_image_occurrences = get_pascal_size_distribution(labels_dir)

    return (object_size_distribution,
            object_size_occurrences,
            size_objects_per_image_distribution,
            size_objects_per_image_occurrences)


def image_aspect_ratio_distribution(images_dir: Path) -> list:
    '''
    Computes the aspect ratio distribution over a images folder.

    :param images_dir: Path of images directory.
    :return: list of occurrences.
    '''
    assert images_dir.exists(), logging.error(f"{images_dir} not found")

    list_aspect_ratio = []

    for extension in LABEL_EXTENSION_LIST:
        if extension == '.xml':
            for file in images_dir.glob(f'*{extension}'):
                im = Image.open(str(file))
                width, height = im.width, im.height
                aspect_ratio = width / height
                list_aspect_ratio.append(aspect_ratio)

    return list_aspect_ratio

if __name__ == '__main__':
    root_dir = Path('../../../Data/FootballerDetection/raw_data')
    print(number_of_images(root_dir))
    print(number_of_labels(root_dir))
    print(class_distribution(root_dir))

