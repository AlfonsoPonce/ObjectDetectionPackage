'''
This module serves auxiliary functions to conversion utils.
Author: Alfonso Ponce Navarro
Date: 31/10/2023
'''

def xml_to_yolo_bbox(bbox, w, h):
    '''
    Converts PASCAL_VOC Bboxes to YOLO ones.
    :param bbox: PASCAL BBox
    :param w: width of BBox, used to normalize
    :param h: height of BBox, used to normalize
    :return: Yolo format BBox
    '''
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def yolo_to_xml_bbox(bbox, w, h):
    '''
    Converts YOLO Bboxes to PASCAL_VOC ones.
    :param bbox: YOLO BBox
    :param w: width of BBox
    :param h: height of BBox
    :return: Pascal format BBox
    '''
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

