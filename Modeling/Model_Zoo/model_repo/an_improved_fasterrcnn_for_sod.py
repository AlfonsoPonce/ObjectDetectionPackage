import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torch import nn, Tensor
from torchvision import ops
from torchvision import _utils
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import AnchorGenerator, RegionProposalNetwork, RoIHeads
from torchvision.models.detection.image_list import ImageList
from typing import Dict

def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

class Improved_Faster_RCNN_For_SOD:


    def __init__(self, num_classes, local_response_norm_size):
        backbone = CustomVGG(local_response_norm_size=local_response_norm_size)
        rpn_head = CustomRegionProposalNetworK()

        self.model = FasterRCNN(backbone=backbone, rpn_head=rpn_head,
        num_classes=num_classes)


class CustomROIHeads(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor,
        # Faster R-CNN training
        fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh, nms_thresh, detections_per_img,):
        super().__init__(box_roi_pool, box_head, box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh, nms_thresh, detections_per_img,)

    def forward(
            self,
            features,  # type: Dict[str, Tensor]
            proposals,  # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
            targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")


        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )





        return result, losses

class CustomRegionProposalNetwork(RegionProposalNetwork):
    def __init__(self,
                 anchor_generator: AnchorGenerator,
                 head: nn.Module,
                 # Faster-RCNN Training
                 fg_iou_thresh: float,
                 bg_iou_thresh: float,
                 batch_size_per_image: int,
                 positive_fraction: float,
                 # Faster-RCNN Inference
                 pre_nms_top_n: Dict[str, int],
                 post_nms_top_n: Dict[str, int],
                 nms_thresh: float,
                 score_thresh: float = 0.0,
                 ):
        super().__init__(anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                             positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh)



    def IIOU(self, predicted_boxes: Tensor, target_boxes: Tensor):
        predicted_bbox_areas = ops.box_area(predicted_boxes)
        target_bbox_areas = ops.box_area(target_boxes)

        lt = torch.max(predicted_boxes[:, None, :2], target_boxes[:, :2])  # [N,M,2]
        rb = torch.min(predicted_boxes[:, None, 2:], target_boxes[:, 2:])  # [N,M,2]

        wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]

        intersection_areas = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        closure_areas = (rb[:, :, 0] - lt[:, :, 0]) * (rb[:, :, 1] - lt[:, :, 1])

        IoU = ops.box_iou(predicted_boxes, target_boxes)

        IIoU = IoU - ((closure_areas - (predicted_bbox_areas + target_bbox_areas - intersection_areas)) / closure_areas)


        return IIoU

    def LIIOU(self, predicted_boxes: Tensor, target_boxes: Tensor):
        return 1 - self.IIOU(predicted_boxes, target_boxes)

    def compute_loss(
            self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        '''
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())
        '''

        box_loss = self.LIIOU(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds]) \
                   / (sampled_inds.numel())

        ##########SOFTMAXLOSS############## REVISAR
        loss = nn.CrossEntropyLoss()
        objectness_loss = loss(objectness[sampled_inds], labels[sampled_inds])
        ##########SOFTMAXLOSS##############



        return objectness_loss, box_loss

    def forward(
            self,
            images: ImageList,
            features: Dict[str, Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = super().concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

class CustomVGG(nn.Module):
    def __init__(self, local_response_norm_size):
        super(CustomVGG, self).__init__()
        self.local_response_norm_size = local_response_norm_size
        self.features = vgg16(weights='DEFAULT').features

        self.out_channels = 512


    def __feature_fusion_map__(self, features_1, features_2, features_3):
        lrn = nn.LocalResponseNorm(self.local_response_norm_size) #This parameter is not specified in paper (size)
        lrn_1 = lrn(features_1)
        lrn_2 = lrn(features_2)
        lrn_3 = lrn(features_3)

        return lrn_1 + lrn_2 + lrn_3

    def __max_pooling_downsampling__(self, input_features: torch.Tensor) -> torch.Tensor:
        pool_layer = nn.MaxPool2d(kernel_size=2, stride=2).to(input_features.device) # To reduce in a half
        channel_augmentation = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1).to(input_features.device) #Make channels be equal
        return channel_augmentation(pool_layer(input_features))

    def __bilinear_upsampling__(self, input_features: torch.Tensor):
        return nn.functional.interpolate(input_features, size=(28, 28), mode='bilinear')



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for i in range(len(self.features)):
            if i == 16:
                conv3_3_output = x #Before max pooling after Relu
            elif i == 23:
                conv4_3_output = x #Before max pooling after Relu
            x = self.features[i](x)
        conv5_3_output = x #After all computations

        downsampled_conv_3_3_output = self.__max_pooling_downsampling__(conv3_3_output)
        upsampled_conv_5_4_output = self.__bilinear_upsampling__(conv5_3_output)

        resultant_output = self.__feature_fusion_map__(downsampled_conv_3_3_output,
                                                       conv4_3_output,
                                                       upsampled_conv_5_4_output)

        return resultant_output








if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    example_image = Path('C:\\Users\\fonso\Documents\\tiles_8\\images\\000069.jpg')

    backbone = CustomVGG(local_response_norm_size=2)
    backbone.to('cuda')
    example_image = Image.open(str(example_image))

    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajustar al tamaño esperado por la red neuronal
        transforms.ToTensor(),  # Convertir la imagen a un tensor
    ])

    image_tensor = transformaciones(example_image).unsqueeze(0)

    #backbone(image_tensor.to('cuda'))

    model = Improved_Faster_RCNN_For_SOD(num_classes=1, local_response_norm_size=2)



