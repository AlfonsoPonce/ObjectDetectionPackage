import wandb
from AnnotationConversor.conversors import voc_to_yolo, voc_to_coco, coco_to_voc
import argparse
import logging
from Augmentations.augment import perform_augmentations
from pathlib import Path
import sys


def run(args):
    log_filename = 'Preprocessing.log'

    # setting logging
    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w')
    logging.root.setLevel(logging.INFO)

    if args.annotation_conversion.upper() == "TRUE":
        run = wandb.init(job_type="Preprocessing_AnnotationConversion")
        run.config.update(args)

        logging.info('Checking valid conversion')
        try:
            assert args.src_label_format != args.dst_label_format
        except AssertionError as error:
            logging.error('Invalid conversion')
            raise error

        # create the labels folder (output directory)
        if not Path.exists(Path(args.annotation_conversor_output_label_dir)):
            Path(
                args.annotation_conversor_output_label_dir).mkdir(
                exist_ok=True)

        if args.src_label_format == 'VOC' and args.dst_label_format == 'YOLO':
            logging.info('Performing VOC =====> YOLO conversion')
            voc_to_yolo(
                args.class_list.split(','), Path(
                    args.input_label_dir), Path(
                    args.annotation_conversor_output_label_dir))

        elif args.src_label_format == 'VOC' and args.dst_label_format == 'COCO':
            logging.info('Performing VOC =====> COCO conversion')
            voc_to_coco(
                args.class_list.split(','), Path(
                    args.input_label_dir), Path(
                    args.annotation_conversor_output_label_dir))

        elif args.src_label_format == 'COCO' and args.dst_label_format == 'VOC':
            logging.info('Performing COCO =====> VOC conversion')
            coco_to_voc(
                Path(
                    args.input_label_dir), Path(
                    args.image_directory), Path(
                    args.annotation_conversor_output_label_dir), 'voc_dataset')

        logging.info('Conversion finished succesfully')

        artifact = wandb.Artifact(
            name=args.annotation_conversor_artifact_name,
            type=args.annotation_conversor_artifact_type,
            description=args.annotation_conversor_artifact_description
        )
        artifact.add_dir(str(Path(args.annotation_conversor_output_label_dir)))
        artifact.add_file(log_filename)
        run.log_artifact(artifact)

    if args.augmentations.upper() == "TRUE":
        run = wandb.init(job_type="Preprocessing_Augmentations")
        run.config.update(args)

        logging.info(
            f"Performing {Path(args.augmentation_file).stem} augmentations...")
        perform_augmentations(
            Path(
                args.image_directory), Path(
                args.labels_directory), Path(
                args.augmentation_file), args.class_list.split(','))
        logging.info(f"Augmentations successfully finished.")

        artifact = wandb.Artifact(
            name=args.augmentations_artifact_name,
            type=args.augmentations_artifact_type,
            description=args.augmentations_artifact_description
        )
        artifact.add_file(str(args.augmentation_file))
        artifact.add_file(log_filename)
        run.log_artifact(artifact)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Module to perform Preprocessing related to Object Detection Datasets')

    parser.add_argument(
        '--annotation_conversion',
        type=str,
        help='If True, annotation conversion will be done with specified options. Else, annotation conversion will not be done',
        required=True)

    parser.add_argument(
        '--class_list',
        type=str,
        help='List of classes to detect. E.g: --class_list Class1,Class2,...,ClassN',
        required=True)

    parser.add_argument(
        '--src_label_format',
        type=str,
        help='Source format of labels',
        choices=[
            'VOC',
            'COCO',
            'YOLO'],
        required=True)

    parser.add_argument(
        '--dst_label_format',
        type=str,
        help='Destination format of labels',
        choices=[
            'VOC',
            'COCO',
            'YOLO'],
        required=True)

    parser.add_argument(
        '--input_label_dir',
        type=str,
        help='Source annotation folder',
        required=True)

    parser.add_argument(
        '--annotation_conversor_output_label_dir',
        type=str,
        help='Destination annotation folder',
        required=True)

    parser.add_argument(
        "--annotation_conversor_artifact_name",
        type=str,
        help="Name of the artifact",
        required=True)

    parser.add_argument(
        "--annotation_conversor_artifact_type",
        type=str,
        help="Type of the artifact",
        required=True)

    parser.add_argument(
        "--annotation_conversor_artifact_description",
        type=str,
        help="A brief description of this artifact",
        required=True)

    parser.add_argument(
        '--augmentations',
        type=str,
        help='If True, augmentations will be done '
        'with specified options. Else, augmentations '
        'will not be done',
        required=True)

    parser.add_argument('--image_directory', type=str,
                        help='Directory to fetch images.', required=True
                        )

    parser.add_argument(
        '--labels_directory',
        type=str,
        help='Directory to fetch labels.',
        required=True)

    parser.add_argument(
        '--augmentation_file',
        type=str,
        help='YAML file with albumentations style',
        required=True)

    parser.add_argument(
        "--augmentations_artifact_name",
        type=str,
        help="Name of the artifact",
        required=True)

    parser.add_argument(
        "--augmentations_artifact_type",
        type=str,
        help="Type of the artifact",
        required=True)

    parser.add_argument(
        "--augmentations_artifact_description",
        type=str,
        help="A brief description of this artifact",
        required=True)

    args = parser.parse_args()
    run(args)
