'''
This Script allows the user to remove a bounding box based on a condition

Author: Alfonso Ponce Navarro
Date: 07/11/2023
'''
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from pascal_voc_writer import Writer
from pathlib import Path
import numpy as np
from xml.dom.minidom import parseString

annotation_dir = Path('/media/getec/DA8CB8FD8CB8D56B/datasets/ANSYS/Delivery/input/')
outpath = Path('/media/getec/DA8CB8FD8CB8D56B/datasets/ANSYS/Delivery/new_labels/')
outpath.mkdir(parents=True, exist_ok=True)

def remove_bbox(condition, annotation_dir: Path, output_dir: Path):
    x = 0
    for file in annotation_dir.glob('*.xml'):

        xmlp = ET.XMLParser(encoding='utf-8')
        tree = ET.parse(file, parser=xmlp)

        root = tree.getroot()

        height = root.find('size').find('height').text
        width = root.find('size').find('width').text
        channels = root.find('size').find('depth').text


        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'VOC2007'
        img_name = root.find('filename').text

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


        for object in root.findall('object'):
            #Si no ocupa toda la image entonces escribimos
            if condition:
                print()
            else:
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = object.find('name').text

                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'

                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = object.find('bndbox').find('xmin').text
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = object.find('bndbox').find('ymin').text
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = object.find('bndbox').find('xmax').text
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = object.find('bndbox').find('ymax').text
                xml = tostring(node_root, pretty_print=True)
                dom = parseString(xml)
        f = open(outpath.joinpath(file.name), "wb")
        f.write(xml)
        f.close()
        print(x)
        x += 1

