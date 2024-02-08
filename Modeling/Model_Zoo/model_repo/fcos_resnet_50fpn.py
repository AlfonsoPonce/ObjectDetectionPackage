import torchvision
import torch
import math
from torchvision.models.detection.fcos import FCOSClassificationHead
from functools import partial


def create_model(num_classes, new_head):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    if new_head:
        num_anchors = model.head.classification_head.num_anchors

        model.head.classification_head = FCOSClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        model.transform.min_size = (640,)
        model.transform.max_size = 640
        for param in model.parameters():
            param.requires_grad = True

    return model

if __name__ == '__main__':
    create_model(2)