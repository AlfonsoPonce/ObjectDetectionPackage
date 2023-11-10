'''
Module that abstracts three types of dataset format: Pascal VOC, COCO and YOLO.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
import torch
from PIL import Image, ImageDraw
import numpy as np
from xml.etree import ElementTree as et
import keyboard
from pathlib import Path
from torch.utils.data import Dataset
from albumentations.core.composition import Compose


# the dataset class
class PascalDataset(Dataset):
    '''
    Class that represents an object detection dataset with Pascal VOC annotation format.
    '''

    def __init__(
            self, image_list: list, labels_list: list, class_list,
            width: int = 0, height: int = 0, transforms: Compose = None
    ):
        '''
        Instantiate a Pascal VOC dataset.

        :param image_list: list of images
        :param labels_list: list of labels
        :param class_list: list of classes,
        :param width: width resize. If no size if given, image is not resized
        :param height: height resize. If no size if given, image is not resized
        :param transforms: list of albumentations transforms.
        '''
        self.transforms = transforms
        self.height = height
        self.width = width
        class_list.insert(0, '__bg__')
        self.class_list = class_list
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        self.all_image_paths = []
        self.all_annot_paths = []
        self.images_path = image_list[0].parent
        self.labels_path = labels_list[0].parent
        # get all the image paths in sorted order

        self.all_image_paths.extend(image_list)

        self.all_annot_paths.extend(labels_list)

        self.all_images = [
            image_path.name for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx: int) -> tuple:
        '''
        Function to return an object detection instance.

        :param idx: instance index.
        :return: Tuple which contains image and bboxes.
        '''
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        # print(image_name)
        image_path = self.images_path.joinpath(image_name)

        # read the image
        image_resized = np.array(
            Image.open(
                str(image_path))).astype(
            np.float32)
        if self.width != 0 and self.height != 0:
            image_resized = image_resized.resize((self.width, self.height))

        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = self.labels_path.joinpath(annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image_resized.shape[1]
        image_height = image_resized.shape[0]

        # box coordinates for xml files are extracted and corrected for image
        # size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.class_list.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            if self.width != 0 and self.height != 0:
                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin = (xmin / image_width) * self.width
                xmax = (xmax / image_width) * self.width
                ymin = (ymin / image_height) * self.height
                ymax = (ymax / image_height) * self.height

            # print(f'ANTES {xmax}---{xmin}' )
            if xmax <= xmin:
                xmax += 0.1
            if ymax <= ymin:
                ymin -= 0.1
            if ymax > image_height:
                ymax = image_height - 1
            if xmax > image_width:
                xmax = image_width - 1
            # if xmin > 1.0: xmin = 1.0
            # if ymin > 1.0: ymin = 1.0
            # print(f'DESPUES {xmax}---{xmin}')
            boxes.append([xmin, ymin, xmax, ymax])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    images_list = list(
        Path('../../Data/FootballerDetection/raw_data/images').glob('*'))
    labels_list = list(
        Path('../../Data/FootballerDetection/raw_data/labels').glob('*'))
    classes = ['player', 'ball', 'goalkeeper', 'referee']
    height = 0
    width = 0

    dataset = PascalDataset(
        images_list, labels_list, classes, width, height
    )
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample

    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = classes[target['labels'][box_num]]
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (int(
                    box[0]), int(
                    box[1]), int(
                    box[2]), int(
                    box[3])), outline=(
                        0, 255, 0))

            draw.text((int(box[0]), int(box[1] - 5)), label)
        image.show()
        keyboard.wait('q')

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(Image.fromarray(np.uint8(image * 255)), target)
