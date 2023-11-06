'''
This module allows the conversion among Object Detection annotations.
Author: Alfonso Ponce Navarro
Date: 31/10/2023
'''
from pathlib import Path
import xml.etree.ElementTree as ET
import glob
import json
from .utils import xml_to_yolo_bbox
import xmltodict

def voc_to_yolo(class_list: list, input_label_dir: Path, output_label_dir: Path):
    '''
    Conversion from PASCAL_VOC annotations to YOLO annotations.
    :param class_list: list of classes.
    :param input_label_dir: Path to the directory containing the annotations.
    :param output_label_dir: Path to the output directory.
    :return: None
    '''


    # identify all the xml files in the annotations folder (input directory)
    file_list = list(input_label_dir.glob('*.xml'))

    # loop through each
    for file in file_list:
        filename = file.stem

        result = []

        # parse the content of the xml file
        try:
            tree = ET.parse(file)
        except:
            xmlp = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(file, parser=xmlp)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in class_list:
                class_list.append(label)
            index = class_list.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]

            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)

            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(output_label_dir.joinpath(f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(class_list))


def voc_to_coco(class_list: list, input_label_dir: Path, output_label_dir: Path):
    """
    Conversion from PASCAL_VOC annotations to COCO annotations
    :param class_list: list of classes
    :param input_label_dir: Path to the directory containing the annotations
    :param output_label_dir: Path to the output directory
    :return: None
    """
    attrDict = dict()
    # images = dict()
    # images1 = list()
    attrDict["categories"] = []
    class_id=1
    for category in class_list:
        attrDict["categories"].append({"supercategory": "none", "id": class_id, "name": category})
        class_id += 1

    images = []
    annotations = []
    image_id = 0

    for annotation_path in input_label_dir.glob('*.xml'):

        image_id += 1

        with annotation_path.open() as xml_file:
            doc = xmltodict.parse(xml_file.read())

        image = {
                    "file_name": doc['annotation']['filename'],
                    "height": int(doc['annotation']['size']['height']),
                    "width": int(doc['annotation']['size']['width']),
                    "id": image_id
                }

        print("File Name: {} and image_id {}".format(annotation_path.name, image_id))
        images.append(image)

        if 'object' in doc['annotation']:
            id1 = 1
            for obj in doc['annotation']['object']:
                for value in attrDict["categories"]:
                    annotation = {}

                    if str(obj['name']) == value["name"]:
                        annotation["iscrowd"] = 0
                        annotation["image_id"] = image_id
                        x1 = int(obj["bndbox"]["xmin"]) - 1
                        y1 = int(obj["bndbox"]["ymin"]) - 1
                        x2 = int(obj["bndbox"]["xmax"]) - x1
                        y2 = int(obj["bndbox"]["ymax"]) - y1
                        annotation["bbox"] = [x1, y1, x2, y2]
                        annotation["area"] = float(x2 * y2)
                        annotation["category_id"] = value["id"]
                        annotation["ignore"] = 0
                        annotation["id"] = id1
                        annotation["segmentation"] = [
                                    [x1, y1, x1, (y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
                        id1 += 1
                        annotations.append(annotation)
        else:
            print("File: {} doesn't have any object".format(annotation_path.name))

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict)
    with open(str(output_label_dir.joinpath('annotations.json')), "w") as f:
        f.write(jsonString)