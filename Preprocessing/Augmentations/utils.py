'''
Module that implements auxiliary augmentation utils.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
from pathlib import Path
from xml.etree import ElementTree as et
import logging


def read_pascal_bboxes(annot_file: Path, class_list: list) -> tuple:
    '''
    Function that reads pascal bboxes and returns them in a tuple

    :param annot_file: label file.
    :param class_list: list of classes to be detected.
    :return: Tuple whose first element are the list of bboxes and the second element is the labels associated with them.
    '''
    assert annot_file.exists(), logging.error(f'{annot_file} not found')

    boxes = []
    labels = []
    tree = et.parse(str(annot_file))
    root = tree.getroot()

    for member in root.findall('object'):
        # map the current object name to `classes` list to get...
        # ... the label index and append to `labels` list
        labels.append(class_list.index(member.find('name').text))

        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])

    return (boxes, labels)
