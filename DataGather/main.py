import wandb
import logging
import argparse
from pretrained_classes import COCO_CLASSES
from Modeling.Inference.dataset_inference import dataset_inference
from Modeling.Model_Zoo.Zoo import Zoo
import torch

from VideoGather.utils import video_sampling
from pathlib import Path
import sys
sys.path.append('../')


def run(args):
    log_filename = 'DataGather.log'

    # setting logging
    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w')
    logging.root.setLevel(logging.INFO)

    logger = logging.getLogger("EDA")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)

    # initiating wandb
    run = wandb.init(job_type="Data_Gather")
    run.config.update(args)

    if args.video_sampling.lower() == 'true':

        dest_folder = Path(args.root_output_folder).joinpath('images')
        dest_folder.mkdir(exist_ok=True, parents=True)
        logging.info(f"Sampled frames will be saved in {str(dest_folder)}")

        logging.info(
            f"Sampling videos from {str(Path(args.input_video_folder))} discarding {float(args.discard_ratio)*100}% of total frames")
        video_sampling(
            Path(
                args.input_video_folder), dest_folder, float(
                args.discard_ratio))

        if args.autolabel.lower() == 'true':
            logging.info(
                f"Executing autolabel with model {args.model_name} pretrained in COCO dataset")
            model_repo = Zoo(len(COCO_CLASSES))
            model = model_repo.get_model(args.model_name, False)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resize = (int(args.resize.split(',')[0]),
                      int(args.resize.split(',')[1]),
                      int(args.resize.split(',')[2]))

            dataset_inference(
                model,
                device,
                dest_folder,
                None,
                dest_folder,
                COCO_CLASSES,
                float(
                    args.detection_threshold),
                resize)

            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description
            )

            artifact.add_dir(str(dest_folder))
            artifact.add_file(log_filename)
            run.log_artifact(artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module for automatic data gathering.')

    parser.add_argument(
        '--video_sampling',
        required=True,
        type=str,
        help='If True, enables sampling frames from video. Sampling'
        'is done randomly.')

    parser.add_argument(
        '--discard_ratio',
        required=True,
        type=str,
        help='Ratio to discard frames when sampling. E.g: If'
        '0.5 is set, Half of produced images will be discarded')

    parser.add_argument(
        '--autolabel',
        required=True,
        type=str,
        help='If True, a model will be used to generate labels. WARNING:'
        'User should be aware of the model used to perform this')

    parser.add_argument(
        '--model_name',
        required=True,
        help='Name of the model. Must be the name of the file under model_repo (without file extension)')

    parser.add_argument(
        '--detection_threshold',
        required=True,
        help='threshold that sets minimum confidence to an object to be detected')

    parser.add_argument(
        '--resize',
        required=True,
        help='Comma separated list of 3 dimensions to resize image')

    parser.add_argument(
        '--input_video_folder',
        required=True,
        help='Path where the video/s is/are stored.')

    parser.add_argument(
        '--root_output_folder',
        required=True,
        help='Folder where results will be stored. If autolabel is set to True,'
        'then a label folder will be produced. If not, just an image folder'
        'will be created.')

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
