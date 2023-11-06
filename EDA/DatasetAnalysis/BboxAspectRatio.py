import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
from multiprocessing import Pool, Queue
import multiprocessing
import math
import itertools
Annotation_Dir = Path('../input/Torch_Ansys_Dataset_v2/train/pascal_labels_v2/')
Class_mean_areas = {}
Class_count = {}
aspect_ratio_list = []
def computeKernel(file_list, q):
    #global aspect_ratio_list
    aspect_ratio_list = []
    for xml_f in file_list:
        try:
            tree = ET.parse(xml_f)
        except:
            xmlp = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(xml_f, parser=xmlp)

        root = tree.getroot()

        for member in root.findall('object'):
            bbox = member.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            object_class = member.find('name').text

            try:
                if Class_mean_areas[object_class] == None:
                    Class_mean_areas[object_class] = 0
                    Class_count[object_class] = 0
            except:
                Class_mean_areas[object_class] = 0
                Class_count[object_class] = 0

            bbox_width = xmax-xmin
            bbox_height = ymax-ymin
            if bbox_width == 0: bbox_width += 0.01
            if bbox_height == 0: bbox_height += 0.01


            #Class_count[object_class] += 1
            #Class_mean_areas[object_class] += Class_mean_areas[object_class]
            aspect_ratio_list.append(round(bbox_width / bbox_height,2))
            #q.put(aspect_ratio_list)
    q.put(aspect_ratio_list)



def getBboxAspectratio():

    bbox_area_list = []
    glob_dir = Annotation_Dir.glob('*.xml')
    file_list = [x for x in glob_dir if x.is_file()]

    num_cpus = multiprocessing.cpu_count()
    pool = Pool(num_cpus)
    lim_inf = 0
    lim_sup = math.floor(len(file_list) / num_cpus)
    batch = lim_sup
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    lists = []
    for i in range(num_cpus):
        results = pool.apply_async(computeKernel, args=(file_list[lim_inf:lim_sup], queue))
        lists.extend(queue.get())

        lim_inf = (lim_sup + 1)
        if math.fabs(lim_sup - len(file_list)) < batch:
            lim_sup += int(math.fabs(lim_sup - len(file_list)))
        else:
            lim_sup += batch

            '''
    for xml_f in file_list:
        try:
            tree = ET.parse(xml_f)
        except:
            xmlp = ET.XMLParser(encoding='utf-8')
            tree = ET.parse(xml_f, parser=xmlp)

        root = tree.getroot()

        for member in root.findall('object'):
            bbox = member.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            object_class = member.find('name').text

            try:
                if Class_mean_areas[object_class] == None:
                    Class_mean_areas[object_class] = 0
                    Class_count[object_class] = 0
            except:
                Class_mean_areas[object_class] = 0
                Class_count[object_class] = 0

            bbox_width = xmax-xmin
            bbox_height = ymax-ymin
            if bbox_width == 0: bbox_width += 0.01
            if bbox_height == 0: bbox_height += 0.01


            Class_count[object_class] += 1
            Class_mean_areas[object_class] += Class_mean_areas[object_class]
            aspect_ratio_list.append(bbox_width / bbox_height)
        '''

    pool.close()
    pool.join()


    #print(len(queue.get()))
    sns.histplot(lists)
    plt.show()


    for classname in list(Class_mean_areas.keys()):
        Class_mean_areas[classname] /= Class_count[classname]

    sns.histplot(x=list(Class_mean_areas.keys()), y=list(Class_mean_areas.values()))
    plt.show()

    sns.histplot(x=list(Class_count.keys()), y=list(Class_count.values()))
    plt.show()

getBboxAspectratio()