from pathlib import Path
import cv2
import numpy as np
import os
from xml.etree import ElementTree as et
import xml.etree.ElementTree as ET
from lxml.etree import tostring
from xml.dom.minidom import parseString

image_folder = Path('../input/Dataset_Final/images')
label_folder = Path('../input/Dataset_Final/pascal_labels')
dest_folder = Path('../Resultado_Reescalado/')
dest_folder_images = Path('../Resultado_Reescalado/images')
dest_folder_labels = Path('../Resultado_Reescalado/pascal_labels')
RESIZE = 224
dest_folder.mkdir(parents=True, exist_ok=True)
dest_folder_images.mkdir(parents=True, exist_ok=True)
dest_folder_labels.mkdir(parents=True, exist_ok=True)

for image_file in image_folder.glob('*jpg'):
        # read the image
        image = cv2.imread(str(image_file))
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (RESIZE, RESIZE))

        cv2.imwrite(str(dest_folder_images) + '\\' + image_file.name, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_file.name[:-4] + '.xml'
        annot_file_path = os.path.join(str(label_folder), annot_filename)

        boxes = []
        labels = []
        try:
            tree = ET.parse(annot_file_path)
        except:
            xmlp = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(annot_file_path, parser=xmlp)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        root.find('size').find('width').text = str(RESIZE)
        root.find('size').find('height').text = str(RESIZE)

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = int((xmin/image_width)*RESIZE)
            xmax_final = int((xmax/image_width)*RESIZE)
            ymin_final = int((ymin/image_height)*RESIZE)
            ymax_final = int((ymax/image_height)*RESIZE)

            ##### HAN HABIDO DATASETS QUE ESTÁN MAL ANOTADOS ####
            if xmax_final <= xmin_final: xmax_final += 0.1
            if ymax_final <= ymin_final: ymin_final -= 0.1
            #if   ymax_final > 1.0: ymax_final = 1.0
            #elif xmax_final > 1.0: xmax_final = 1.0
            #elif xmin_final > 1.0: xmin_final = 1.0
            #elif ymin_final > 1.0: ymin_final = 1.0
            ##### HAN HABIDO DATASETS QUE ESTÁN MAL ANOTADOS ####

            member.find('bndbox').find('xmin').text = str(xmin_final)
            member.find('bndbox').find('xmax').text = str(xmax_final)
            member.find('bndbox').find('ymin').text = str(ymin_final)
            member.find('bndbox').find('ymax').text = str(ymax_final)

        xml = ET.tostring(root, encoding='utf-8', method='xml')
        dom = parseString(xml)

        f = open(str(dest_folder_labels.joinpath(annot_filename)), "wb")
        f.write(xml)
        f.close()



        print(f'{image_file} processed')