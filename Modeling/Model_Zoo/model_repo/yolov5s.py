from ultralytics import YOLO
import torch
def create_model(num_classes, new_head):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=num_classes, pretrained=True, autoshape=False)

    if new_head == True:
        print(model)

    return model

if __name__ == '__main__':
    create_model(2, True)