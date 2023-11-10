'''
Class that represents model storage.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
import torch

from .model_repo import fasterrcnn_mobilenetv3_large_fpn, fasterrcnn_resnet50, fcos_resnet_50fpn, \
    retinanet_resnet50_fpn, ssd_vgg16


class Zoo():
    def __init__(self, num_classes: int):
        '''
        Instantiate Zoo object.
        :param num_classes: number of classes to be detected
        '''
        # add background class
        self.num_classes = num_classes + 1

    def get_model(self, name: str):
        '''
        Function to select detection model
        :param name: name of the model to be used. The name must be the same as the filename in model_repo/ folder.
        :return: torch object detection.
        '''
        model = None
        if name == 'fasterrcnn_resnet50':
            model = fasterrcnn_resnet50.create_model(self.num_classes)

        elif name == 'fasterrcnn_mobilenetv3_large_fpn':
            model = fasterrcnn_mobilenetv3_large_fpn.create_model(
                self.num_classes)

        elif name == 'fcos_resnet_50fpn':
            model = fcos_resnet_50fpn.create_model(self.num_classes)

        elif name == 'retinanet_resnet_50fpn':
            model = retinanet_resnet50_fpn.create_model(self.num_classes)

        elif name == 'ssd_vgg16':
            model = ssd_vgg16.create_model(self.num_classes)

        return model
