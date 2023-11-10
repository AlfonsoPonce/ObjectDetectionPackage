'''
Module that implements augmentations utilites.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''

from .utils import read_pascal_bboxes
import math
import albumentations as A
from PIL import Image
from pathlib import Path
from pascal_voc_writer import Writer
from multiprocessing import Pool
import multiprocessing
import numpy as np
import logging


def perform_augmentations(
        image_directory: Path,
        labels_directory: Path,
        augmentations_file: Path,
        class_list: list) -> None:
    '''
    Function to do image augmentations using multiprocessing.

    :param image_directory: Directory to fetch images.
    :param labels_directory: Directory to fetch labels.
    :param augmentations_file: YAML file with albumentations style.
    :param class_list: list of classes
    '''
    try:
        assert image_directory.exists()
    except AssertionError as err:
        logging.error(f"{image_directory} not found.")
        raise err

    try:
        assert labels_directory.exists()
    except AssertionError as err:
        logging.error(f"{labels_directory} not found.")
        raise err

    try:
        assert augmentations_file.exists()
    except AssertionError as err:
        logging.error(f"{augmentations_file} not found.")
        raise err

    Augmented_Image_Dir = image_directory.joinpath(augmentations_file.stem)
    Augmented_Labels_Dir = labels_directory.joinpath(augmentations_file.stem)
    if not Path.exists(Augmented_Image_Dir):
        Augmented_Image_Dir.mkdir(exist_ok=True)
    if not Path.exists(Augmented_Labels_Dir):
        Augmented_Labels_Dir.mkdir(exist_ok=True)
    transforms = A.load(str(augmentations_file), data_format='yaml')

    # list_dir = os.listdir(image_directory)

    list_dir = list(image_directory.glob('*'))

    num_cpus = multiprocessing.cpu_count()
    pool = Pool(int(num_cpus / 2))
    lim_inf = 0
    lim_sup = math.floor(len(list_dir) / num_cpus)
    batch = lim_sup

    for i in range(num_cpus):
        pool.apply_async(compute_kernel,
                         args=(image_directory,
                               labels_directory,
                               class_list,
                               list_dir[lim_inf:lim_sup],
                               transforms,
                               Augmented_Image_Dir,
                               Augmented_Labels_Dir))
        lim_inf = (lim_sup + 1)
        if math.fabs(lim_sup - len(list_dir)) < batch:
            lim_sup += int(math.fabs(lim_sup - len(list_dir)))
        else:
            lim_sup += batch

    pool.close()
    pool.join()


def compute_kernel(
        image_directory: Path,
        labels_directory: Path,
        class_list: list,
        list_dir: list,
        transforms: object,
        Augmented_Image_Dir: Path,
        Augmented_Labels_Dir: Path) -> None:
    '''
    Core function that performs image augmentations.

    :param image_directory: Directory to fetch images.
    :param labels_directory: Directory to fetch labels.
    :param class_list: list of classes
    :param list_dir: list of images to augment
    :param transforms: Albumentations object that performs augmentation
    :param Augmented_Image_Dir: Directory to store augmented images
    :param Augmented_Labels_Dir: Directory to store augmented labels.
    '''
    list_dir2 = [x.stem for x in list_dir if x.is_file()]

    i = 0
    for image_file in list_dir:

        if image_file.is_file():
            label_file = labels_directory.joinpath(
                image_file.name.replace(image_file.suffix, '.xml'))

            image = np.array(Image.open(str(image_file)))
            if image is not None:
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                bboxes, labels = read_pascal_bboxes(
                    image, label_file, class_list)

                for x in range(len(labels)):
                    labels[x] = class_list[labels[x]]

                try:
                    transformed = transforms(
                        image=image, bboxes=bboxes, class_labels=labels)

                    transformed_image = Image.fromarray(transformed['image'])
                    transformed_class_labels = transformed['class_labels']
                    transformed_bboxes = transformed['bboxes']

                    # cv2.imwrite(str(Augmented_Image_Dir.joinpath(str(list_dir2[i]) + ".jpg")), cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
                    transformed_image.save(
                        str(Augmented_Image_Dir.joinpath(image_file.name)))

                    if transforms.get_dict_with_id(
                    )['bbox_params']['format'] == 'pascal_voc':

                        writer = Writer(str(image_directory.joinpath(
                            image_file.name)), transformed_image.width, transformed_image.height)

                        for x in range(len(transformed_class_labels)):
                            xmin = transformed_bboxes[x][0]
                            ymin = transformed_bboxes[x][1]
                            xmax = transformed_bboxes[x][2]
                            ymax = transformed_bboxes[x][3]

                            writer.addObject(transformed_class_labels[x], int(
                                xmin), int(ymin), int(xmax), int(ymax))
                        print(image_file)
                        writer.save(
                            str(Augmented_Labels_Dir.joinpath(image_file.stem + ".xml")))
                    print("Image" +
                          str(list_dir2[i]) +
                          ".png processed and saved")

                except Exception as e:
                    # print(e)
                    raise
                i += 1


if __name__ == '__main__':
    perform_augmentations(
        Path('../../Data/FootballerDetection/raw_data/images'),
        Path('../../Data/FootballerDetection/raw_data/labels'),
        Path('./transformations/RandomRain.yml'),
        [
            'referee',
            'player',
            'ball',
            'goalkeeper'])
