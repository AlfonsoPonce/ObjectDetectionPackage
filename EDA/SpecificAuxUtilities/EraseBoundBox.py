import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from pascal_voc_writer import Writer
from pathlib import Path
import numpy as np
from xml.dom.minidom import parseString

annotation_dir = Path('/media/getec/DA8CB8FD8CB8D56B/datasets/ANSYS/Delivery/input/')
outpath = Path('/media/getec/DA8CB8FD8CB8D56B/datasets/ANSYS/Delivery/new_labels/')
outpath.mkdir(parents=True, exist_ok=True)

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
        if (object.find('bndbox').find('xmin').text == str(0) and object.find('bndbox').find('ymin').text == str(0)) and object.find('bndbox').find('xmax').text == str(width) and object.find('bndbox').find('ymax').text == str(height):
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



    '''
    target = (annopath % img_id)
    if os.path.exists(target):
        label_norm = np.loadtxt(target).reshape(-1, 5)

        for i in range(len(label_norm)):
            labels_conv = label_norm[i]
            new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3],
                                  labels_conv[4])
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = classes[new_label[0]]

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
    f = open(outpath % img_id, "wb")
    # f = open(os.path.join(outpath, img_id), "w")
    # os.remove(target)
    f.write(xml)
    f.close()
    '''
