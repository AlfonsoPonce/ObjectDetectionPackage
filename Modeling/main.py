from pathlib import Path
import random
import math
from Model_Zoo.Zoo import Zoo
from Dataset.dataset import PascalDataset
from Training.train import train
from Training.custom_utils import collate_fn, get_train_transform, get_valid_transform
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import argparse
import json

def run(args):


    if args.dataset_format == 'VOC':
        extension = '.xml'
    images_list = list(Path(args.images_dir).glob('*'))

    train_images_list = random.sample(images_list, math.floor(args.train_split * len(images_list)))
    valid_images_list = [test_image for test_image in images_list if test_image not in train_images_list]

    train_labels_list = [Path(args.labels_dir).joinpath(file.with_suffix(extension).name) for file in train_images_list]
    valid_labels_list = [Path(args.labels_dir).joinpath(file.with_suffix(extension).name) for file in valid_images_list]

    train_dataset = PascalDataset(train_images_list, train_labels_list, args.class_list.split(','), transforms=get_train_transform())
    valid_dataset = PascalDataset(valid_images_list, valid_labels_list, args.class_list.split(','), transforms=get_valid_transform())

    train_dataloader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)


    pretrained_models = Zoo(len(args.class_list.split(',')))
    model = pretrained_models.get_model(args.model)

    train_config_dict = json.loads(args.train_config.replace('\'', '\"'))

    optimizer_name = train_config_dict['optimizer']['name']
    optimizer_params = train_config_dict['optimizer']['params']
    optimizer_params['params'] = model.parameters()
    train_config_dict['optimizer'] = getattr(optim, optimizer_name)(**optimizer_params)

    scheduler_name = train_config_dict['scheduler']['name']
    scheduler_params = train_config_dict['scheduler']['params']
    scheduler_params['optimizer'] = train_config_dict['optimizer']
    train_config_dict['scheduler'] = getattr(scheduler, scheduler_name)(**scheduler_params)

    train(model, train_config_dict, train_dataloader, valid_dataloader, Path(args.output_dir))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Module to perform model training, optimization, testing...')

    parser.add_argument('--dataset_format', type=str,
                        help='Label format. It can be VOC, COCO or YOLO', choices=['VOC', 'COCO', 'YOLO'],
                        required=True)

    parser.add_argument('--class_list', type=str,
                        help='List of classes to detect. E.g: --class_list Class1,Class2,...,ClassN',
                        required=True)

    parser.add_argument('--images_dir', type=str, help='Folder where images are stored',
                         required=True)

    parser.add_argument('--labels_dir', type=str, help='Folder where labels are stored', required=True)

    parser.add_argument('--output_dir', type=str, help='Output directory for results', required=True)

    parser.add_argument('--batch_size', type=int, help='Batch size',
                        required=True)

    parser.add_argument("--train_split", type=float, help="Ratio of training instances", required=True)

    parser.add_argument("--model", type=str, help="Name of the model. Must be the name of the file under model_repo (without file extension)", required=True)

    parser.add_argument(
        "--train_config", type=str, help="Training configuration dictionary. Must be given a string formatted with json.",
        required=True
    )


    parser.add_argument("--artifact_name", type=str, help="Name of the artifact", required=True)

    parser.add_argument("--artifact_type", type=str, help="Type of the artifact", required=True)

    parser.add_argument(
        "--artifact_description", type=str, help="A brief description of this artifact", required=True
    )

    args = parser.parse_args()
    run(args)

