from pathlib import Path
import xml.etree.ElementTree as ET
from lxml.etree import tostring
from xml.dom.minidom import parseString
from pascal_voc_writer import Writer
import cv2
image_folder = Path('../input/BarcosCiviles/')
label_folder = Path('../input/BarcosCiviles/')

label_folder.mkdir(exist_ok=True)

for file in image_folder.glob('*.jpg'):
    xml_file = str(file).replace('images', 'pascal_labels').replace('jpg', 'xml')
    image = cv2.imread(str(file))
    orig_width = image.shape[1]
    orig_height = image.shape[0]
    writer = Writer(file, orig_width, orig_height)

    xmin = orig_width - 20
    ymin = orig_height - 20
    xmax = orig_width
    ymax = orig_height

    writer.addObject('Ruido', xmin, ymin, xmax, ymax)

    writer.save(xml_file)