import argparse
from pathlib import Path
from ExtraPreprocessing.extra_label_eraser import erase_extra_labels
from ExtraPreprocessing.label_rename import label_rename
from ExtraPreprocessing.erase_images_with_empty_label import remove_images_with_empty_labels
def run(args):

    if args.type.lower() == "remove_extra_labels":
        to_remain_list = [label.replace('_', ' ') for label in args.labels_to_maintain.split(',')]

        erase_extra_labels(Path(args.labels_directory), to_remain_list)

    elif args.type.lower() == "rename_labels":
        current_labels_list = [label.replace('_', ' ') for label in args.current_labels_list.split(',')]
        target_labels_list = [label.replace('_', ' ') for label in args.target_labels_list.split(',')]

        label_rename(Path(args.labels_directory), current_labels_list, target_labels_list)

    elif args.type.lower() == "erase_images_with_empty_labels":
        remove_images_with_empty_labels(Path(args.image_directory), Path(args.labels_directory))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Module to perform extra preprocessing steps')

    parser.add_argument('--type', type=str, required=True, help='Type of extra preprocessing steps',
                        choices=['remove_extra_labels', 'rename_labels', 'erase_images_with_empty_labels'])

    parser.add_argument('--image_directory', type=str, required=True, help='Path where images are stored.')

    parser.add_argument('--labels_directory', type=str, required=True, help='Path where labels are stored.')

    parser.add_argument('--labels_to_maintain', type=str, required=True, help='comma separated list of labels to keep. The rest will be removed.')

    parser.add_argument('--current_labels_list', type=str, required=True, help='comma separated list of current labels to be renamed.')

    parser.add_argument('--target_labels_list', type=str, required=True, help='comma separated list of new labels.')

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