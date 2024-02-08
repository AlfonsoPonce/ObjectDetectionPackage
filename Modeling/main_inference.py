'''
Main module that implements inference functionality

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''

from Model_Zoo.Zoo import Zoo
import torch
from Inference.video_inference import local_video_inference
from Inference.dataset_inference import dataset_inference
from pathlib import Path
import argparse
from Inference.CLASS_LIST import COCO_CLASSES, CUSTOM_CLASSES

def run(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.checkpoint.lower() == 'none':
        model_repo = Zoo(len(COCO_CLASSES))

        model = model_repo.get_model(args.model_name, False)

        class_list = COCO_CLASSES
    else:
        model_repo = Zoo(len(CUSTOM_CLASSES))

        model = model_repo.get_model(args.model_name, True)

        checkpoint = torch.load(str(Path(args.checkpoint)), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        class_list = CUSTOM_CLASSES

    if args.inference_type.lower() == 'video':
        local_video_inference(Path(args.input_folder),
                              Path(args.output_folder),
                              model,
                              class_list,
                              float(args.detection_threshold),
                              device,
                              (int(args.resize.split(',')[0]),
                               int(args.resize.split(',')[1])))
    elif args.inference_type.lower() == 'dataset':
        dataset_inference(model, device, Path(args.input_folder), None,
                          Path(args.output_folder),class_list, float(args.detection_threshold),
                          (int(args.resize.split(',')[0]),
                           int(args.resize.split(',')[1])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference module')


    parser.add_argument('--inference_type', required=True, choices=['video', 'dataset'],
                        help='Type of data which inference will be done.')

    parser.add_argument('--input_folder', required=True, help='Folder where input data is stored.')

    parser.add_argument('--output_folder', required=True, help='Folder output data will be stored')



    parser.add_argument('--model_name', required=True,
                        help='Name of the model. Must be the name of the file under model_repo (without file extension)')

    parser.add_argument('--checkpoint', required=True,
                        help='Name of model checkpoint. If no checkpoint is used, type None.')

    parser.add_argument('--detection_threshold', required=True,
                        help='threshold that sets minimum confidence to an object to be detected')

    parser.add_argument('--resize', required=True, help='Comma separated list of 2 dimensions to resize image')

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name of the artifact",
        required=True)

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the artifact",
        required=True)

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="A brief description of this artifact",
        required=True)

    args = parser.parse_args()

    run(args)
