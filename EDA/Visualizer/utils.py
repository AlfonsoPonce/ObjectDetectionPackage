'''
This module serves utilities to visualize pascal voc datasets.
Date: 27/01/2024
Author: Alfonso Ponce Navarro
'''
from pathlib import Path
import cv2
from xml.etree import ElementTree as ET
from tqdm import tqdm
import logging


def voc_visualizer(root_input_dir: Path, root_output_dir: Path) -> None:
    '''
    Function that saves Pascal Voc annotated images in a folder.

    :param root_input_dir: Root path of folder where images and labels subfolders are stored.
    :param root_output_dir: Root path of folder where annotated images will be shown.

    '''

    try:
        assert root_input_dir.exists()
    except AssertionError as err:
        logging.error(f"{str(root_input_dir)} not found.")
        raise err

    root_output_dir.mkdir(exist_ok=True, parents=True)

    image_list = list(root_input_dir.joinpath('images').glob('*.*'))

    for image_file in tqdm(image_list, desc='Processing images...'):

        image = cv2.imread(str(image_file))
        image = cv2.putText(image, image_file.name, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        label_file = root_input_dir.joinpath(
            'labels', f'{image_file.stem}.xml')

        annotation_list = get_voc_annotations(label_file)

        for ann in annotation_list:
            bbox_width = ann[1]['xmax'] - ann[1]['xmin']
            bbox_height = ann[1]['ymax'] - ann[1]['ymin']
            bbox_area = bbox_height * bbox_width

            box_color = (0, 255, 0)  # Green

            image = cv2.rectangle(
                image,
                (ann[1]['xmin'],
                 ann[1]['ymin']),
                (ann[1]['xmax'],
                 ann[1]['ymax']),
                box_color,
                1)
            image = cv2.putText(
                image,
                ann[0] +
                str(bbox_area),
                (ann[1]['xmin'],
                 ann[1]['ymin']),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,
                 0,
                 0),
                1)

        cv2.imwrite(str(root_output_dir.joinpath(image_file.name)), image)


def get_voc_annotations(label_file: Path) -> list:
    '''
    Gets all Bbox annotations in a pascal voc file.

    :param label_file: pascal VOC label file
    :return: list of all bboxes in label_file
    '''
    try:
        tree = ET.parse(label_file)
    except BaseException:
        xmlp = ET.XMLParser(encoding='utf-8')
        tree = ET.parse(label_file, parser=xmlp)

    root = tree.getroot()

    annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        annotations.append(
            [class_name, {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}])

    return annotations


if __name__ == '__main__':
    root_dir = Path('C:\\Users\\fonso\\Documents\\sample_output\\tiles')
    dst_dir = Path('C:\\Users\\fonso\\Documents\\sample_output\\tiles\\result')

    voc_visualizer(root_dir, dst_dir)
