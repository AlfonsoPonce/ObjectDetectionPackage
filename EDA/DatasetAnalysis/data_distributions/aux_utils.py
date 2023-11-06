'''
This module performs some auxiliary functions such as reading from labels.

Author: Alfonso Ponce Navarro
Date: 25/10/2023
'''

import xml.etree.ElementTree as ET
from pathlib import Path

def get_pascal_class_distribution(labels_dir: Path) -> tuple:
    '''
    Computes the class distribution among PASCAL VOC format labels.
    :param labels_dir (Path): Directory where labels are stored
    :return (Tuple): First element is the distribution dictionary. Second element is a list with all class appearances.
    '''

    ocurrence = []
    class_distribution = {}
    for xml_file in labels_dir.glob(f'*.xml'):
        xmlp = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(xml_file, parser=xmlp)
        root = tree.getroot()

        for member in root.findall('object'):
            class_name = member[0].text
            ocurrence.append(class_name)
            if class_name not in list(class_distribution.keys()):
                class_distribution[class_name] = 1
            else:
                class_distribution[class_name] += 1

    return class_distribution, ocurrence




def get_pascal_size_distribution(labels_dir: Path) -> tuple:
    '''
    Computes the distribution of objects sizes in the PASCAL VOC dataset format. Besides, returns how many images contain
    small, medium and large objects.
    :param label_dir (Path): Directory where labels are stored
    :return (Tuple): First element is the object size distribution. Second element is the number of images with different
                    object sizes
    '''

    object_size_distribution = {'small': 0, 'medium': 0, 'large': 0}
    size_objects_per_image_distribution = {'small': 0, 'medium': 0, 'large': 0}
    object_size_occurences = []
    size_objects_per_image_occurrences = []

    for file in labels_dir.glob('*.xml'):
        image_has_small_object = False
        image_has_medium_object = False
        image_has_large_object = False
        try:
            tree = ET.parse(file)
        except:
            xmlp = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(file, parser=xmlp)

        root = tree.getroot()

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin, ymin = int(bbox.find('xmin').text), int(bbox.find('ymin').text)
            xmax, ymax = int(bbox.find('xmax').text), int(bbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin
            area = width * height

            if area < 1024:
                image_has_small_object = True
                object_size_distribution['small'] += 1
                object_size_occurences.append('small')
            elif 1024 <= area and area < 4096:
                image_has_medium_object = True
                object_size_distribution['medium'] += 1
                object_size_occurences.append('medium')
            else:
                image_has_large_object = True
                object_size_distribution['large'] += 1
                object_size_occurences.append('large')


        if image_has_small_object:
            size_objects_per_image_distribution['small'] += 1
            size_objects_per_image_occurrences.append('small')
        if image_has_medium_object:
            size_objects_per_image_distribution['medium'] += 1
            size_objects_per_image_occurrences.append('medium')
        if image_has_large_object:
            size_objects_per_image_distribution['large'] += 1
            size_objects_per_image_occurrences.append('large')




    return (object_size_distribution,
            object_size_occurences,
            size_objects_per_image_distribution,
            size_objects_per_image_occurrences)



'''
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax
def isSmallObject(bbox, w, h):
    x1, y1, x2, y2 = yolo2bbox(bbox)
    # denormalize the coordinates
    xmin = int(x1 * w)
    ymin = int(y1 * h)
    xmax = int(x2 * w)
    ymax = int(y2 * h)
    width = xmax - xmin
    height = ymax - ymin

    area = width*height

    if area < 1024:
        return True
    else:
        return False



def yoloSmallObjects(ruta_bboxes):
    total_objects = 0
    small_objects = 0
    images_with_small_object = 0
    total_images = 0

    for file in os.listdir(ruta_bboxes):
        has_small_object = False
        with open(ruta_bboxes+file, "r") as f:
            for lines in f:
                lines = lines.replace("\n", "")
                lines = lines.replace(" ", ", ")
                lines = lines[3:]
                lines = [float(i) for i in lines.split(',')]
                if isSmallObject(lines, image_width, image_height):
                    small_objects += 1
                    has_small_object = True
                total_objects += 1
        if has_small_object:
            images_with_small_object += 1
        total_images += 1
'''