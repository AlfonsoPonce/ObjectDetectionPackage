import torchvision
import torch
import math
from torchvision.models.detection.fcos import FCOSHead


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    num_anchors = model.head.classification_head.num_anchors
    out_channels = model.head.classification_head.conv[9].out_channels
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(
        out_channels,
        num_anchors *
        num_classes,
        kernel_size=3,
        stride=1,
        padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    # as per pytorcch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits

    return model
