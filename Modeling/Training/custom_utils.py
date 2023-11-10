'''
Module that implements some custom utilities

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''
import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from albumentations.pytorch import ToTensorV2


plt.style.use('ggplot')


class Averager:
    '''
    This class keeps track of the training and validation loss values
    and helps to get the average for each epoch as well
    '''

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch) -> tuple:
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def get_train_transform() -> A.Compose:
    '''
    Define the training transforms.

    :return: training transformation
    '''
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_valid_transform() -> A.Compose:
    '''
    Define the validation transforms.

    :return: validation transformation
    '''
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def save_model(save_dir: Path, name: str, epoch: int, model,
               optimizer: torch.optim.Optimizer) -> None:
    """
    Function to save the trained model till current epoch, or whenever called.

    :param epoch: The epoch number.
    :param name: name of the trained model.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    """
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_dir.joinpath(f"{name}.pth"))


def save_train_loss_plot(
        save_dir: Path,
        name: str,
        train_loss_list: list) -> None:
    """
    Function to save both train loss graph.

    :param save_dir: Path to save the graphs.
    :param name: name of file to save.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure_1.savefig(f"{save_dir}/{name}.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')


def draw_boxes(image, box: list, color, resize=None):
    """
    This function will annotate images with bounding boxes
    based on wether resizing was applied to the image or not.

    :param image: Image to annotate.
    :param box: Bounding boxes list.
    :param color: Color to apply to the bounding box.
    :param resize: Either None, or provide a Tuple (width, height)


    Returns:
           image: The annotate image.
    """
    if resize is not None:
        cv2.rectangle(image,
                      (
                          int((box[0] / resize[0]) * image.shape[1]),
                          int((box[1] / resize[1]) * image.shape[0])
                      ),
                      (
                          int((box[2] / resize[0]) * image.shape[1]),
                          int((box[3] / resize[1]) * image.shape[0])
                      ),
                      color, 2)
        return image
    else:
        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      color, 2)
        return image


def put_class_text(image, box, class_name: str, color, resize: tuple = None):
    """
    Annotate the image with class name text.

    :param image: The image to annotate.
    :param box: List containing bounding box coordinates.
    :param class_name: Text to put on bounding box.
    :param color: Color to apply to the text.
    :param resize: Whether annotate according to resized coordinates or not.

    Returns:
           image: The annotated image.
    """
    if resize is not None:
        cv2.putText(image, class_name,
                    (
                        int(box[0] / resize[0] * image.shape[1]),
                        int(box[1] / resize[1] * image.shape[0] - 5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                    2, lineType=cv2.LINE_AA)
        return image
    else:
        cv2.putText(image, class_name,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                    2, lineType=cv2.LINE_AA)
        return image
