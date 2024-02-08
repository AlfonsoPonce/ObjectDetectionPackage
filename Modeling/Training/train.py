'''
Module that implements model training

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''

from .train_utils.engine import train_one_epoch, evaluate

from pathlib import Path
from .custom_utils import (
    save_model,
    save_train_loss_plot,
    Averager
)
import torch
import logging

# if __name__ == '__main__':
def train(
        model,
        train_config_dict: dict,
        train_loader,
        valid_loader,
        output_dir: Path) -> tuple:
    '''
    Function that implements model training.

    :param model: torch detection model
    :param train_config_dict: configuration training dictionary
    :param train_loader: torch train dataloader
    :param valid_loader: torch valid dataloader
    :param output_dir: Path where trained model will be stored
    :return: Best and last metrics.
    '''
    try:
        assert isinstance(train_config_dict['optimizer'], torch.optim.Optimizer)
    except AssertionError as err:
        logging.error("Torch optimizer is not Pytorch object")
        raise err
    try:
        assert isinstance(train_config_dict['epochs'], int)
    except AssertionError as err:
        logging.error("Number of epoch is not int")
        raise err

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(valid_loader.dataset)}\n")

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = train_config_dict['optimizer']
    scheduler = train_config_dict['scheduler']
    num_epochs = train_config_dict['epochs']
    BEST_MAP = (0, 0)

    for epoch in range(num_epochs):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler
        )

        model_name = model.__class__.__name__

        cocoEval = evaluate(model, valid_loader, device=device)

        map_05_095 = cocoEval.stats[0].item()
        map_05 = cocoEval.stats[1].item()
        mar_05_095 = cocoEval.stats[6].item()

        print(map_05_095)
        print(map_05)

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)

        # Save the current epoch model.
        save_model(output_dir, f'LAST_{model_name}', epoch, model, optimizer)

        if map_05_095 > BEST_MAP[1]:
            save_model(
                output_dir,
                f'BEST_{model_name}',
                epoch,
                model,
                optimizer)
            BEST_MAP = (map_05, map_05_095)

        # Save loss plot.
        save_train_loss_plot(output_dir, model_name, train_loss_list)

    # shutil.copyfile("config_file.json", out + f"/config_file.json")

    LAST_MAP = (cocoEval.stats[1].item(), cocoEval.stats[0].item())

    return BEST_MAP, LAST_MAP



if __name__ == '__main__':
    from ..Model_Zoo.Zoo import Zoo

    model_repo = Zoo(num_classes=1)
    model = model_repo.get_model('fasterrcnn_resnet50', True)

    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    class_list = ['ball']
    from ..Dataset.dataset import PascalDataset
    from .custom_utils import collate_fn, get_train_transform, get_valid_transform
    from torch.utils.data import DataLoader

    images_dir = 'C:\\Users\\fonso\\Documents\\sample_output\\subset\\images'
    labels_dir = 'C:\\Users\\fonso\\Documents\\sample_output\\subset\\labels'

    images_list = list(Path(images_dir).glob('*'))
    import random
    import math

    train_config_dict = {
        'batch_size': 1,
        'epochs': 10,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    train_split = 0.8
    train_images_list = random.sample(
        images_list, math.floor(
            train_split * len(images_list)))
    valid_images_list = [
        test_image for test_image in images_list if test_image not in train_images_list]

    train_labels_list = [
        Path(
            labels_dir).joinpath(
            file.with_suffix('.xml').name) for file in train_images_list]
    valid_labels_list = [
        Path(
            labels_dir).joinpath(
            file.with_suffix('.xml').name) for file in valid_images_list]

    train_dataset = PascalDataset(
        train_images_list,
        train_labels_list,
        class_list,
        transforms=get_train_transform())
    valid_dataset = PascalDataset(
        valid_images_list,
        valid_labels_list,
        class_list,
        transforms=get_valid_transform())

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config_dict["batch_size"],
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=train_config_dict["batch_size"],
        collate_fn=collate_fn)


    train(model= model,
          train_config_dict=train_config_dict,
          train_loader=train_dataloader,
          valid_loader=valid_dataloader,
          output_dir=Path('./prueba'))