'''
Module that implements auxiliary augmentation utils.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
import os
import pathlib
from xml.etree import ElementTree as et
import albumentations as A


def boxTypeDetection(Annot_Directory):
    annot_list = os.listdir(Annot_Directory)
    isYOLO = False
    isPascal = False
    isCOCO = False

    for file in annot_list:
        extension = file.split('.')[1]
        if extension == 'txt':
            isYOLO = isYOLO or True
        elif extension == 'json':
            isCOCO = isCOCO or True
        elif extension == 'xml':
            isPascal = isPascal or True

    if isYOLO and not isCOCO and not isPascal:
        return 'yolo'
    elif isCOCO and not isYOLO and not isPascal:
        return 'coco'
    elif isPascal and not isYOLO and not isCOCO:
        return 'pascal_voc'
    else:
        raise 'More than one file format in the folder'

def getClassList(input):
    string_list = input.split(',')
    list = []
    iterator = 1
    for element in string_list:
        if element == ' ':
            return False, 'Class list can not have blank spaces'
        if iterator % 2 == 0 and element != ',':
            return False, 'Class list must be separated by commas'

        list.append(element)
    return True, list

#def readYOLOBboxes(annot_file):



def readPascalBboxes(image, annot_file, classes):

    boxes = []
    labels = []
    tree = et.parse(str(annot_file))
    root = tree.getroot()

    for member in root.findall('object'):
        # map the current object name to `classes` list to get...
        # ... the label index and append to `labels` list
        labels.append(classes.index(member.find('name').text))

        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)


        boxes.append([xmin, ymin, xmax, ymax])


    return boxes, labels


#def readCocoBboxes(annot_file):