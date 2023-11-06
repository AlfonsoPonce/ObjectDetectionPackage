'''
This module serves data distributions in number of classes and relative object sizes.

Date: 25/10/2023
Author: Alfonso Ponce Navarro
'''
from pathlib import Path

from .aux_utils import get_pascal_class_distribution, get_pascal_size_distribution

IMAGE_EXTENSION_LIST = ['.png', '.jpg', '.jpeg']
LABEL_EXTENSION_LIST = ['.xml', '.json', '.txt']

def number_of_images(images_dir: Path) -> int:
    '''
    Computes the number of images in a data folder. Images must be in images/ folder below task/ folder.
    :param root_dir (Path): Root directory of the data
    :return (int): Number of images in data
    '''
    all_images = []
    for extension in IMAGE_EXTENSION_LIST:
        all_images.extend(images_dir.glob(f'*{extension}'))

    return len(all_images)

def number_of_labels(labels_dir: Path) -> int:
    '''
    Computes the number of labels in a data folder. Images must be in labels/ folder below task/ folder.
    :param root_dir (Path): Root directory of the data
    :return (int): Number of images in data
    '''
    all_labels = []
    for extension in LABEL_EXTENSION_LIST:
        all_labels.extend(labels_dir.glob(f'*{extension}'))

    return len(all_labels)

def class_distribution(labels_dir: Path) -> dict:
    '''
    Computes class distribution among labels
    :param root_dir (Path): Root directory of the data
    :return (Tuple): First element is the distribution dictionary. Second element is a list with all class appearances.
    '''

    for extension in LABEL_EXTENSION_LIST:
        if extension == '.xml':
            distribution_dict, class_ocurrence_list = get_pascal_class_distribution(labels_dir)

    return distribution_dict, class_ocurrence_list

def relative_object_size_distribution(labels_dir: Path) -> tuple:
    '''

    :param root_dir (Path): Root directory of the data
    :return (Tuple): First element is the object size distribution. Second element is the number of images with different
                    object sizes
    '''


    for extension in LABEL_EXTENSION_LIST:
        if extension == '.xml':
            object_size_distribution, object_size_occurrences, \
            size_objects_per_image_distribution, size_objects_per_image_occurrences = get_pascal_size_distribution(labels_dir)

    return (object_size_distribution,
            object_size_occurrences,
            size_objects_per_image_distribution,
            size_objects_per_image_occurrences)




if __name__ == '__main__':
    root_dir = Path('../../../Data/FootballerDetection/raw_data')
    print(number_of_images(root_dir))
    print(number_of_labels(root_dir))
    print(class_distribution(root_dir))

