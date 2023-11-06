import argparse
import sys
import os
from pylabel import importer
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict
from yolo_to_voc import xml_transform
def fromYolo2Pascal(annotatios_path, img_path, dst_path, class_list, dataset_name):
    """
    Conversion from Yolo annotations to Pascal annotations
    :param annotatios_path: Path to the directory containing the annotations
    :param img_path: Path to the directory containing the images
    :param dst_path: Path were converted annotations will be put
    :param class_list: List of strings with classes. It must be ordered according to their annotations.
                       If we have ["class1", "class2", "class3"] then class1 = 0, class2 = 1 and class3 = 2
    :param dataset_name: Name for the dataset. It can be whatever
    :return:
    """
    dataset = importer.ImportYoloV5(path=annotatios_path, path_to_images=img_path, cat_names=class_list,
                                    img_ext='jpg', name=dataset_name)
    dataset.export.ExportToVoc()
    os.system('mv ' +annotatios_path + '*.xml ' + dst_path )
    #os.rename(annotatios_path + dataset_name + '.xml', dst_path + '/' + dataset_name + '.xml')

def fromCoCo2Pascal(annotatios_path, img_path, dst_path, dataset_name):
    """
    Conversion from Yolo annotations to Pascal annotations
    :param annotatios_path: Path to the directory containing the annotations
    :param img_path: Path to the directory containing the images
    :param dst_path: Path were converted annotations will be put
    :param dataset_name: Name for the dataset. It can be whatever
    :return:
    """
    dataset = importer.ImportCoco(path=annotatios_path, path_to_images=img_path , name=dataset_name)
    dataset.export.ExportToVoc(output_path=dst_path)
    #os.system('mv ' +annotatios_path + '*.xml ' + dst_path )
    #os.rename(annotatios_path + dataset_name + '.xml', dst_path + '/' + dataset_name + '.xml')

def fromPascal2Coco(annotatios_path, img_path, dst_path, dataset_name):
    """
    Conversion from Yolo annotations to Pascal annotations
    :param annotatios_path: Path to the directory containing the annotations
    :param img_path: Path to the directory containing the images
    :param dst_path: Path were converted annotations will be put
    :param dataset_name: Name for the dataset. It can be whatever
    :return:
    """
    #dataset = importer.ImportVOC(path=annotatios_path, path_to_images=img_path , name=dataset_name)
    XMLFiles = list()
    for file in os.listdir(annotatios_path):
        if file.endswith(".xml"):
            XMLFiles.append(file)
    generateVOC2Json(annotatios_path, XMLFiles)
    #os.system('mv ' +annotatios_path + '*.xml ' + dst_path )
    #os.rename(annotatios_path + dataset_name + '.xml', dst_path + '/' + dataset_name + '.xml')



def generateVOC2Json(rootDir, xmlFiles):
    attrDict = dict()
    # images = dict()
    # images1 = list()
    attrDict["categories"] = [{"supercategory": "none", "id": 1, "name": "header"},
                              {"supercategory": "none", "id": 2, "name": "row"},
                              {"supercategory": "none", "id": 3, "name": "logo"},
                              {"supercategory": "none", "id": 4, "name": "item_name"},
                              {"supercategory": "none", "id": 5, "name": "item_desc"},
                              {"supercategory": "none", "id": 6, "name": "price"},
                              {"supercategory": "none", "id": 7, "name": "total_price_text"},
                              {"supercategory": "none", "id": 8, "name": "total_price"},
                              {"supercategory": "none", "id": 9, "name": "footer"}
                              ]
    images = list()
    annotations = list()
    for root, dirs, files in os.walk(rootDir):
        image_id = 0
        for file in xmlFiles:
            image_id = image_id + 1
            if file in files:

                # image_id = image_id + 1
                annotation_path = os.path.abspath(os.path.join(root, file))

                # tree = ET.parse(annotation_path)#.getroot()
                image = dict()
                # keyList = list()
                doc = xmltodict.parse(open(annotation_path).read())
                # print doc['annotation']['filename']
                image['file_name'] = str(doc['annotation']['filename'])
                # keyList.append("file_name")
                image['height'] = int(doc['annotation']['size']['height'])
                # keyList.append("height")
                image['width'] = int(doc['annotation']['size']['width'])
                # keyList.append("width")

                # image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
                image['id'] = image_id
                print("File Name: {} and image_id {}".format(file, image_id))
                images.append(image)
                # keyList.append("id")
                # for k in keyList:
                # 	images1.append(images[k])
                # images2 = dict(zip(keyList, images1))
                # print images2
                # print images

                # attrDict["images"] = images

                # print attrDict
                # annotation = dict()
                id1 = 1
                if 'object' in doc['annotation']:
                    for obj in doc['annotation']['object']:
                        for value in attrDict["categories"]:
                            annotation = dict()
                            # if str(obj['name']) in value["name"]:
                            if str(obj['name']) == value["name"]:
                                # print str(obj['name'])
                                # annotation["segmentation"] = []
                                annotation["iscrowd"] = 0
                                # annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
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
                    print("File: {} doesn't have any object".format(file))
            # image_id = image_id + 1

            else:
                print("File: {} not found".format(file))

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    # print attrDict
    jsonString = json.dumps(attrDict)
    with open("annotations.json", "w") as f:
        f.write(jsonString)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', choices=['yolo', 'coco', 'pascal_voc'],type=str, help='original annotation format')
    parser.add_argument('--to', choices=['yolo', 'coco', 'pascal_voc'], type=str, help='new annotation format')
    parser.add_argument('--annotations_path', type=str, help='directory where annotations are found, in case of COCO, it must end with the filename')
    parser.add_argument('--img_path', type=str, help='directory where images are found')
    parser.add_argument('--dst_path', type=str, help='path to save the converted annotations')
    parser.add_argument('--dataset_name', type=str, help='name of the dataset')
    parser.add_argument('--class_list', type=list, help='list of classes')

    opt = parser.parse_args()



    if opt.original == 'yolo' and opt.to == 'pascal_voc':
        xml_transform(opt)
    elif opt.original == 'coco' and opt.to == 'pascal_voc':
        fromCoCo2Pascal(opt.annotations_path, opt.img_path, opt.dst_path, opt.dataset_name)
    elif opt.original == 'pascal_voc' and opt.to == 'coco':
        fromPascal2Coco(opt.annotations_path, opt.img_path, opt.dst_path, opt.dataset_name)
    
