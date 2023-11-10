'''
This module allows the conversion among Object Detection annotations.
Author: Alfonso Ponce Navarro
Date: 31/10/2023
'''
from pathlib import Path
import xml.etree.ElementTree as ET
import glob
import json
from .utils import xml_to_yolo_bbox, unconvert
import xmltodict
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from PIL import Image
import logging
from pylabel import importer


def voc_to_yolo(
        class_list: list,
        input_label_dir: Path,
        output_label_dir: Path) -> None:
    '''
    Conversion from PASCAL_VOC annotations to YOLO annotations.

    :param class_list: list of classes.
    :param input_label_dir: Path to the directory containing the annotations.
    :param output_label_dir: Path to the output directory.
    '''
    assert input_label_dir.exists(), logging.error(
        f"{input_label_dir} not found")
    assert output_label_dir.exists(), logging.error(
        f"{output_label_dir} not found")

    # identify all the xml files in the annotations folder (input directory)
    file_list = list(input_label_dir.glob('*.xml'))

    # loop through each
    for file in file_list:
        filename = file.stem

        result = []

        # parse the content of the xml file
        try:
            tree = ET.parse(file)
        except BaseException:
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


def voc_to_coco(
        class_list: list,
        input_label_dir: Path,
        output_label_dir: Path) -> None:
    '''
    Conversion from PASCAL_VOC annotations to COCO annotations.

    :param class_list: list of classes.
    :param input_label_dir: Path to the directory containing the annotations.
    :param output_label_dir: Path to the output directory.
    '''
    assert input_label_dir.exists(), logging.error(
        f"{input_label_dir} not found")
    assert output_label_dir.exists(), logging.error(
        f"{output_label_dir} not found")

    attrDict = dict()
    # images = dict()
    # images1 = list()
    attrDict["categories"] = []
    class_id = 1
    for category in class_list:
        attrDict["categories"].append(
            {"supercategory": "none", "id": class_id, "name": category})
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

        print(
            "File Name: {} and image_id {}".format(
                annotation_path.name,
                image_id))
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
            print(
                "File: {} doesn't have any object".format(
                    annotation_path.name))

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict)
    with open(str(output_label_dir.joinpath('annotations.json')), "w") as f:
        f.write(jsonString)


def yolo_to_voc(
        input_images_dir: Path,
        input_label_dir: Path,
        output_label_dir: Path,
        class_list: list) -> None:
    '''
    Conversion from YOLO annotations to PASCAL VOC annotations.

    :param input_images_dir: Folder where images are stored.
    :param input_label_dir: Folder where input yolo labels are stored.
    :param output_label_dir: Folder where ouput voc labels will be stored.
    :param class_list: list of classes to be detected.
    '''
    assert input_images_dir.exists(), logging.error(
        f"{input_images_dir} not found")
    assert input_label_dir.exists(), logging.error(
        f"{input_label_dir} not found")
    assert output_label_dir.exists(), logging.error(
        f"{output_label_dir} not found")

    yolo_labels_list = list(input_label_dir.glob('*.txt'))

    ids = [file.stem for file in yolo_labels_list]

    for image_file in input_images_dir.glob('*'):

        img = Image.open(str(image_file))
        height, width = img.height, img.width
        channels = len(img.mode)

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'VOC2007'

        img_name = image_file.name
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name

        node_source = SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'Coco database'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = input_label_dir.joinpath(image_file.stem + '.txt')
        if target.exists():
            label_norm = np.loadtxt(target).reshape(-1, 5)

            for idx_label in range(len(label_norm)):
                labels_conv = label_norm[idx_label]
                new_label = unconvert(
                    labels_conv[0],
                    width,
                    height,
                    labels_conv[1],
                    labels_conv[2],
                    labels_conv[3],
                    labels_conv[4])

                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = class_list[new_label[0]]

                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'

                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[1])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[3])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[4])
                xml = tostring(node_root, pretty_print=True)
                dom = parseString(xml)
        # print(xml)
        f = open(str(output_label_dir.joinpath(image_file.stem + '.xml')), "wb")
        # f = open(os.path.join(outpath, img_id), "w")
        # os.remove(target)
        f.write(xml)
        f.close()


def coco_to_voc(
        input_label_path: Path,
        img_path: Path,
        output_label_path: Path,
        dataset_name: str) -> None:
    """
    Conversion from COCO annotations to PASCAL VOC annotations.

    :param annotatios_path: Path to the directory containing the annotations.
    :param img_path: Path to the directory containing the images.
    :param dst_path: Path were converted annotations will be stored.
    :param dataset_name: Name for the dataset. It can be whatever.
    """
    assert img_path.exists(), logging.error(f"{img_path} not found")
    assert input_label_path.exists(), logging.error(
        f"{input_label_path} not found")
    assert output_label_path.exists(), logging.error(
        f"{output_label_path} not found")

    dataset = importer.ImportCoco(
        path=str(input_label_path),
        path_to_images=str(img_path),
        name=dataset_name)
    dataset.export.ExportToVoc(output_path=str(output_label_path))
