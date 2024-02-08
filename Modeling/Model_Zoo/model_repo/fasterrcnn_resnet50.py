import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes, new_head):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    print('=================================')

    if new_head:
        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        for param in model.parameters():
            param.requires_grad = True
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.\n")

    return model




if __name__ == '__main__':
    create_model(4, True)