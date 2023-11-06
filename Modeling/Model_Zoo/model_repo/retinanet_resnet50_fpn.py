import torchvision
from torchvision.models.detection.retinanet import RetinaNetHead
import torch
import math

def create_model(num_classes):
    # load Faster RCNN pre-trained model
    #model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, num_classes=num_classes)

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)



    #DESPUES DE DEPURAR
    num_anchors = 9
    in_features = 256
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits

    return model

#create_model(5)