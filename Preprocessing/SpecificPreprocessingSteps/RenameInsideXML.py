from pathlib import Path
import xml.etree.ElementTree as ET
from lxml.etree import tostring
from xml.dom.minidom import parseString

from multiprocessing import Pool
import multiprocessing
import math
import time


def computeKernel(list_dir):
    for file in list_dir:
        try:
            tree = ET.parse(file)
        except:
            xmlp = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(file, parser=xmlp)

        root = tree.getroot()

        path = root.find('path').text
        split_path = path.split('/')
        file = split_path[-1]

        filename = root.find('filename').text
        new_filename = file.name.replace('xml', 'png')

        root.find('filename').text = new_filename

        xml = ET.tostring(root, encoding='utf-16', method='xml')
        dom = parseString(xml)

        f = open(file, "wb")
        f.write(xml)
        f.close()






path_to_xml = Path('../input/Dataset_BarcosV6/train/pascal_labels')
#all_files = path_to_xml.glob('*.xml')
#list_dir = [file for file in all_files if file.is_file()]

i=1
for file in path_to_xml.glob('*.xml'):
    try:
        tree = ET.parse(file)
    except:
        xmlp = ET.XMLParser(encoding='utf-8')
        tree = ET.parse(file, parser=xmlp)

    root = tree.getroot()
    image_width = root.find('size').find('width').text
    image_height = root.find('size').find('height').text

    #path = root.find('filename').text
    #split_path = path.split('/')
    #file = split_path[-1]

    #new_path = root.find('path').text
    #new_filename = file.name.replace('xml', 'jpg')
    #root.find('path').text = new_path.replace('images', 'labels')
    #root.find('filename').text = new_filename
    #root.find('folder').text = 'labels'

    for label in root.findall('object'):
        if label.find('name').text == 'Submartino':
            print(file.name)
            label.find('name').text = 'Submarino'
            #label.find('bndbox').find('xmin').text = str(int(image_width) - 20)
            #label.find('bndbox').find('ymin').text = str(int(image_height)-20)
            #label.find('bndbox').find('xmax').text = image_width
            #label.find('bndbox').find('ymax').text = image_height
            #label.find('truncated').text = '0'

    '''
    if root.findall('object') == []:
        object = ET.Element('object')
        name = ET.Element('name')
        pose = ET.Element('pose')
        truncated = ET.Element('trucated')
        difficult = ET.Element('difficult')
        bndbox = ET.Element('bndbox')
        xmin = ET.Element('xmin')
        ymin = ET.Element('ymin')
        xmax = ET.Element('xmax')
        ymax = ET.Element('ymax')

        name.text = 'Ruido'
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'
        xmin.text = '1'
        ymin.text = '1'
        xmax.text = image_width
        ymax.text = image_height

        root.append(object)
        object.append(name)
        object.append(pose)
        object.append(truncated)
        object.append(bndbox)
        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)
    '''
    '''
    else:
        for label in root.findall('object'):
            if label.find('name').text == 'Ruido':
                label.find('name').text = 'Ruido'
                label.find('bndbox').find('xmin').text = '1'
                label.find('bndbox').find('ymin').text = '1'
                label.find('bndbox').find('xmax').text = image_width
                label.find('bndbox').find('ymax').text = image_height
                label.find('truncated').text = '0'
    '''
    xml = ET.tostring(root, encoding='utf-8', method='xml')
    dom = parseString(xml)

    f = open(file, "wb")
    f.write(xml)
    f.close()
    print(i)
    i+=1

'''
num_cpus = multiprocessing.cpu_count()
pool = Pool(num_cpus)
lim_inf = 0
lim_sup = math.floor(len(list_dir) / num_cpus)
batch = lim_sup
start = time.time()
for i in range(num_cpus):
    pool.apply_async(computeKernel, args=(list_dir[lim_inf:lim_sup]))
    lim_inf = (lim_sup + 1)
    if math.fabs(lim_sup - len(list_dir)) < batch:
        lim_sup += int(math.fabs(lim_sup - len(list_dir)))
    else:
        lim_sup += batch

    print(list_dir[lim_inf:lim_sup])


pool.close()
pool.join()
print(time.time()-start)
'''

