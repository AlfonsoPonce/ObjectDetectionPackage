import math

import numpy
from shapely import Polygon
from pathlib import Path
from PIL import Image
from xml.etree import ElementTree as ET


from image_bbox_slicer.slicer import Slicer

def tiler(root_src_path: Path, root_dst_path: Path, slice_size: int, label_format: str) -> None:
    '''
    This function converts images into
    blocks of slice_size x slice_size
    '''

    try:
        assert (label_format.upper() in ['VOC', 'YOLO', 'COCO'])
    except AssertionError as err:
        print(f"INVALID LABEL FORMAT: {label_format.upper()}")
        print(f"VALID FORMATS: {['VOC', 'YOLO', 'COCO']}")
        raise err
    label_extensions_dict = {'VOC': '.xml', 'YOLO': '.txt', 'COCO': '.json'}

    root_dst_path.joinpath('images').mkdir(exist_ok=True, parents=True)
    root_dst_path.joinpath('labels').mkdir(exist_ok=True, parents=True)

    for imname in list(root_src_path.joinpath('images').glob('*')):
        #im = cv2.imread(imname)
        im = Image.open(imname)
        height, width = im.height, im.width
        #height, width, _ = im.shape
        h_new = math.ceil(height / slice_size) * slice_size
        w_new = math.ceil(width / slice_size) * slice_size
        #im = cv2.resize(im, (w_new, h_new), cv2.INTER_LINEAR)
        im = im.resize((w_new, h_new), Image.BILINEAR)
        #labname = imname.replace(ext, '.txt')
        #labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        labname = imname.parents[1].joinpath('labels', imname.name.replace(imname.suffix, label_extensions_dict[label_format]))
        if label_format == 'VOC':
            box_list = get_voc_boxes(labname, width, height, w_new, h_new)


        # create tiles and find intersection with bounding boxes for each tile
        for i in range((h_new // slice_size)):
            for j in range((w_new // slice_size)):
                x1 = j * slice_size
                y1 = h_new - (i * slice_size)
                x2 = ((j + 1) * slice_size) - 1
                y2 = (h_new - (i + 1) * slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in box_list:
                    #if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])
                        if not imsaved:

                            sliced_im = numpy.array(im)[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]

                            dst_slice_label = root_dst_path.joinpath('labels', f'{imname.stem}_{i}_{j}{label_extensions_dict[label_format]}')
                            dst_img = root_dst_path.joinpath('images', f'{imname.stem}_{i}_{j}{imname.suffix}')

                            Image.fromarray(sliced_im).save(str(dst_img))
                            imsaved = True

                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope

                        # get central point for the new bounding box
                        centre = new_box.centroid

                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy

                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size

                        if label_format == 'VOC':
                            get_sliced_bbox_voc()

                if len(slice_labels) > 0:
                    if label_format == 'VOC':
                        save_sliced_voc(labname)




    print("tiling successfully completed")

def get_voc_boxes(label_file: Path, width: int, height: int, w_new: int, h_new: int) -> list:
    # parse the content of the xml file
    try:
        tree = ET.parse(label_file)
    except BaseException:
        xmlp = ET.XMLParser(encoding='utf-8')
        tree = ET.parse(label_file, parser=xmlp)

    root = tree.getroot()

    box_list = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        xmin = math.ceil(int(obj.find('bndbox').find('xmin').text) * w_new / width)
        ymin = math.ceil(int(obj.find('bndbox').find('ymin').text) * h_new / height)
        xmax = math.ceil(int(obj.find('bndbox').find('xmax').text) * w_new / width)
        ymax = math.ceil(int(obj.find('bndbox').find('ymax').text) * h_new / height)


        box_list.append((class_name, Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])))

    return box_list

def get_sliced_bbox_voc():
    return

def save_sliced_voc(dst_label_file: Path, ):

    return

'''
def yolo_new_bbox():

    # we have to normalize central x and invert y for yolo format
    new_x = (centre.coords.xy[0][0] - x1) / slice_size
    new_y = (y1 - centre.coords.xy[1][0]) / slice_size
    slice_labels.append([box[0], new_x, new_y, new_width, new_height])
'''

'''
def yolo_method(label_file: Path) -> list:
    # we need to rescale coordinates from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * w_new
    labels[['y1', 'h']] = labels[['y1', 'h']] * h_new

    box_list = []

    # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
    for row in labels.iterrows():
        x1 = row[1]['x1'] - row[1]['w'] / 2
        y1 = (h_new - row[1]['y1']) - row[1]['h'] / 2
        x2 = row[1]['x1'] + row[1]['w'] / 2
        y2 = (h_new - row[1]['y1']) + row[1]['h'] / 2

        box_list.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
        
    return box_list
'''

if __name__ == '__main__':
    root_src_path = Path('C:\\Users\\fonso\\Documents\\sample_output\\subset')
    root_dst_path = Path('C:\\Users\\fonso\\Documents\\tiles_8')
    slice_size = 224
    label_format = 'VOC'


    slicer = Slicer()
    slicer.config_dirs(img_src=str(root_src_path.joinpath('images')),
                       img_dst=str(root_dst_path.joinpath('images')),
                       ann_src=str(root_src_path.joinpath('labels')),
                       ann_dst=str(root_dst_path.joinpath('labels')))


    slicer.keep_partial_labels = False #En caso de que el objeto se parta, se desecha la bbox

    slicer.ignore_empty_tiles = True

    slicer.save_before_after_map = True

    slicer.slice_by_number(number_tiles=8)
    slicer.visualize_sliced_random()



    #tiler(root_src_path, root_dst_path, slice_size, label_format)