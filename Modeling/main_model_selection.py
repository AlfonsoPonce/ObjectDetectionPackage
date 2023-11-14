import argparse
import logging
import wandb
from pathlib import Path
from Model_Selection.selection_methods import k_fold_cross_validation
from Model_Selection.comparison_methods import pairwise_ttest
from Model_Zoo.Zoo import Zoo
import json

def run(args):
    log_filename = 'Model_Selection.log'

    # setting logging
    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w')
    logging.root.setLevel(logging.INFO)

    run = wandb.init(job_type="Model Selection")
    run.config.update(args)

    logging.info("Creating training and validation datasets...")
    if args.dataset_format == 'VOC':
        extension = '.xml'

    try:
        assert Path(args.images_dir).exists()
    except AssertionError as err:
        logging.error(f"{Path(args.images_dir)} not found.")
        raise err

    try:
        assert Path(args.labels_dir).exists()
    except AssertionError as err:
        logging.error(f"{Path(args.images_dir)} not found.")
        raise err

    logging.info("Getting model from repo...")
    pretrained_models = Zoo(len(args.class_list.split(',')))
    train_config_dict = json.loads(args.train_config.replace('\'', '\"'))
    models_results_list = []
    if args.selection_method.upper() == 'K_CROSSVAL':
        for curr_model in args.model_list.split(','):
            model = pretrained_models.get_model(curr_model)

            curr_result_list = k_fold_cross_validation(int(args.K), model, train_config_dict, args.class_list.split(','), Path(args.images_dir), Path(args.labels_dir), Path(args.output_dir))
            models_results_list.append(curr_result_list)


    if args.hypothesis_test.upper == "PAIRWISE_TTEST":
        if len(models_results_list) == 2:
            pairwise_ttest(models_results_list[0], models_results_list[1])
        elif len(models_results_list) > 2:
            #TODO apply multiple pairwise tests with correction



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Module to perform model selection.')
    parser.add_argument(
        '--images_dir',
        type=str,
        help='Folder where images are stored',
        required=True)

    parser.add_argument(
        '--labels_dir',
        type=str,
        help='Folder where labels are stored',
        required=True)
    parser.add_argument(
        '--dataset_format',
        type=str,
        help='Label format. It can be VOC, COCO or YOLO',
        choices=[
            'VOC',
            'COCO',
            'YOLO'],
        required=True)

    parser.add_argument(
        '--class_list',
        type=str,
        help='List of classes to detect. E.g: --class_list Class1,Class2,...,ClassN',
        required=True)

    parser.add_argument(
        "--train_config",
        type=str,
        help="Training configuration dictionary. Must be given a string formatted with json.",
        required=True)

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for results',
        required=True)

    parser.add_argument('--model_list', type=str, help='Comma-separated list (no whitespaces) of models to be compared.'
                                                       'Each model name must be the name of the file under model_repo (without file extension)', required=True)
    parser.add_argument('--metric_critera', type=str, help='Selects which metric will be used to perform comparison', required=True, choices=['MAP'])
    parser.add_argument('--K', type=str, help='Selects K for K_Crossvalidation if applied.')

    parser.add_argument('--selection_method', type=str, help='Method to perform model comparison and selection. If more than'
                                                             'two models are listed, and `pairwise comparison method is specified,'
                                                             'then it will perform all possible comparisons.', required=True,
                        choices=['K_Crossval'])
    parser.add_argument('--hypothesis_test', type=str, help='Hypothesis test to perform model comparison.', required=True)

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