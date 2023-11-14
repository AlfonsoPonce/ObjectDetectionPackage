'''
Module that serves Model Selection methods.

Author: Alfonso Ponce Navarro
Date: 14/11/2023
'''

from sklearn.model_selection import KFold
from pathlib import Path
from ..Dataset.dataset import PascalDataset
from torch.utils.data import DataLoader
from ..Training.custom_utils import collate_fn, get_train_transform, get_valid_transform
from ..Training.train import train

def k_fold_cross_validation(K, model,
        train_config_dict: dict,
        class_list : list,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path) -> list:
    '''
    Perform K cross validation over object detection datasets.

    :param K: Number of splits in data
    :param model: torch detection model
    :param train_config_dict: configuration training dictionary
    :param class_list: list of classes to be detected.
    :param images_dir: Path to stored images
    :param labels_dir: Path to stored labels
    :param output_dir: Path to where results are going to be stored
    :return: a list with the best obtained metrics over all the splits.
    '''

    k_folds = KFold(n_splits=K)

    images_list = list(images_dir.glob('*'))
    labels_list = list(labels_dir.glob('*'))

    result_list = []

    for i, (train_index, valid_index) in enumerate(k_folds.split(images_list)):
        train_dataset = PascalDataset(
            images_list[train_index],
            labels_list[train_index],
            class_list,
            transforms=get_train_transform())

        valid_dataset = PascalDataset(
            images_list[valid_index],
            labels_list[valid_index],
            class_list,
            transforms=get_valid_transform())

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=train_config_dict["batch_size"],
                                      collate_fn=collate_fn)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=train_config_dict["batch_size"],
            collate_fn=collate_fn)

        best_result, last_result = train(model, train_config_dict, train_dataloader, valid_dataloader, output_dir)

        result_list.append(best_result)

    return result_list
