import logging
import wandb
from Visualizer.utils import voc_visualizer
import sys
import argparse
from pathlib import Path


def run(args):
    log_filename = 'Visualizer.log'

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
    run = wandb.init(job_type="Visualization")
    run.config.update(args)

    if args.dataset_format.upper() == 'VOC':
        logging.info(
            f"Visualizing VOC dataset in {str(Path(args.root_output_dir))}")
        voc_visualizer(Path(args.root_input_dir), Path(args.root_output_dir))

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )

    artifact.add_dir(str(Path(args.root_output_dir)))
    artifact.add_file(log_filename)
    run.log_artifact(artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module to perform annotation visualization')

    parser.add_argument(
        '--dataset_format',
        help='Format of dataset, allowed values are [VOC, YOLO, COCO]',
        choices=[
            'VOC',
            'YOLO',
            'COCO'],
        type=str,
        required=True)

    parser.add_argument(
        '--root_input_dir',
        help='Root path of folder where images and labels subfolders are stored. ',
        type=str,
        required=True)

    parser.add_argument(
        '--root_output_dir',
        help='Root path of folder where annotated images will be shown.',
        type=str,
        required=True)

    parser.add_argument("--artifact_name", type=str,
                        help="Name of the artifact", required=True)

    parser.add_argument("--artifact_type", type=str,
                        help="Type of the artifact", required=True)

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="A brief description of this artifact",
        required=True)

    args = parser.parse_args()

    run(args)
