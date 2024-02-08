import sys
import logging
import argparse
from pathlib import Path
from DatasetAnalysis.report_creation import Report
from DatasetAnalysis.data_checks.utils import get_images_shapes, check_label_extensions, check_images_extensions
from DatasetAnalysis.data_distributions.utils import number_of_images, number_of_labels,\
    class_distribution, relative_object_size_distribution
from DatasetAnalysis.data_distributions.plots import plot_histogram
import wandb


def run(args):
    log_filename = 'EDA.log'

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
    run = wandb.init(job_type="Generate_Report")
    run.config.update(args)

    # setting directories
    images_output_dir = Path(args.plots_output_dir)
    save_dir = Path(args.report_save_dir)
    Images_dir = Path(args.images_dir)
    Labels_dir = Path(args.labels_dir)

    logging.info(
        "Checking if output directory for images exists, if not, it will be created...")
    if not images_output_dir.exists():
        logging.info("Directory not found, creating it...")
        images_output_dir.mkdir(exist_ok=True)
        logging.info("Directory created!")

    logging.info(
        "Checking if output directory for storing the report exists, if not, it will be created...")
    if not save_dir.exists():
        logging.info("Directory not found, creating it...")
        save_dir.mkdir(exist_ok=True)
        logging.info("Directory created!")

    logging.info("Creating document...")
    document = Report("Exploratory Data Analysis in Object Detection")

    ######### PREAMBLE #########
    text = "This is an automatic-generated report showing an Exploratory Data Analysis over " \
           "an Object Detection Dataset. Every text, data and image in this document has been put " \
           "through Python programming."
    document.add_section("PREAMBLE", text)
    ######### PREAMBLE #########

    ######### INTRODUCTION #########
    text = "The dataset analyzed contains images of football matches, whose annotations belong to" \
           "specific football elements such as referees, players, etc."
    document.add_section("INTRODUCTION", text)
    ######### INTRODUCTION #########

    ######### SHAPE OF IMAGES #########
    logging.info("Checking images shapes...")
    results = get_images_shapes(Images_dir)
    text = f"Found shapes are: {results}"
    logging.info("Adding results to document...")
    document.add_section("SHAPE OF IMAGES", text)
    ######### SHAPE OF IMAGES #########

    ######### LABELS EXTENSIONS #########
    logging.info("Checking labels extension...")
    results = check_label_extensions(Labels_dir)
    text = f"Found extensions are: {results}"
    logging.info("Adding results to document...")
    document.add_section("LABELS EXTENSIONS", text)
    ######### LABELS EXTENSIONS #########

    ######### IMAGES EXTENSIONS #########
    logging.info("Checking images extension...")
    results = check_images_extensions(Images_dir)
    text = f"Found extensions are: {results}"
    logging.info("Adding results to document...")
    document.add_section("IMAGES EXTENSIONS", text)
    ######### IMAGES EXTENSIONS #########

    ######### UNPAIRED IMAGES-LABELS #########
    logging.info("Checking unpaired entities...")
    results = check_label_extensions(Labels_dir)
    text = f"Unpaired entities: {results}"
    logging.info("Adding results to document...")
    document.add_section("UNPAIRED ENTITIES", text)
    ######### UNPAIRED IMAGES-LABELS #########

    ######### NUMBER OF IMAGES #########
    logging.info("Getting number of images...")
    results = number_of_images(Images_dir)
    text = f"Number of images: {results}"
    logging.info("Adding results to document...")
    document.add_section("NUMBER OF IMAGES", text)
    ######### NUMBER OF IMAGES #########

    ######### NUMBER OF LABELS #########
    logging.info("Getting number of labels...")
    results = number_of_labels(Labels_dir)
    text = f"Number of images: {results}"
    logging.info("Adding results to document...")
    document.add_section("NUMBER OF LABELS", text)
    ######### NUMBER OF LABELS #########

    ######### DISTRIBUTION OF CLASSES #########
    logging.info("Computing class distribution...")
    results = class_distribution(Labels_dir)
    text = f"Class distribution: {results[0]}"
    logging.info("Adding results to document...")
    document.add_section("CLASS DISTRIBUTION", text)

    file_name = str(Path.joinpath(
        Path.cwd(), images_output_dir, Path("ClassDistribution.png")))
    logging.info("Adding Class Distribution Plot...")
    plot_histogram(results[1], file_name)
    text = f"Class Distribution Histogram"
    logging.info("Adding results to document...")
    document.add_plot(350, text, file_name)
    ######### DISTRIBUTION OF CLASSES #########

    ######### DISTRIBUTION OF OBJECT SIZES #########
    logging.info("Computing object size distribution...")
    results = relative_object_size_distribution(Labels_dir)
    text = f"Object size distribution: {results[0]}"
    logging.info("Adding results to document...")
    document.add_section("OBJECT SIZE DISTRIBUTION", text)

    file_name = str(Path.joinpath(Path.cwd(), images_output_dir,
                    Path("ObjectSizeDistribution.png")))
    logging.info("Adding Object Size Distribution Plot...")

    plot_histogram(results[1], file_name)
    text = f"Object Size Distribution Histogram"
    logging.info("Adding results to document...")
    document.add_plot(350, text, file_name)
    ######### DISTRIBUTION OF OBJECT SIZES #########

    ######### DISTRIBUTION OF OBJECT SIZES IN ALL IMAGES #########
    logging.info("Computing object size distribution...")
    results = relative_object_size_distribution(Labels_dir)
    text = f"Object size distribution: {results[2]}"
    logging.info("Adding results to document...")
    document.add_section("DISTRIBUTION OF OBJECT SIZES IN ALL IMAGES", text)

    file_name = str(Path.joinpath(Path.cwd(), images_output_dir,
                    Path("NumberOfImagesWithObjectSize.png")))
    logging.info("Adding Object Size Distribution Plot...")

    plot_histogram(results[3], file_name)
    text = f"Object Size Distribution Histogram"
    logging.info("Adding results to document...")
    document.add_plot(350, text, file_name)
    ######### DISTRIBUTION OF OBJECT SIZES IN ALL IMAGES #########

    try:
        logging.info("Saving document...")
        document.save_document(save_dir)
    except Exception as err:
        logging.warning("Following happened during document saving:")
        logging.error(str(err))


    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )

    artifact.add_file(str(Path.joinpath(save_dir, "EDA_REPORT.pdf")))
    artifact.add_file(log_filename)
    run.log_artifact(artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform EDA on object detection dataset")

    parser.add_argument("--images_dir", type=str,
                        help="Image folder to analyse", required=True)

    parser.add_argument("--labels_dir", type=str,
                        help="Label folder to analyse", required=True)

    parser.add_argument(
        "--plots_output_dir",
        type=str,
        help="Directory to save generated images",
        required=True)

    parser.add_argument(
        "--report_save_dir",
        type=str,
        help="Directory where report is going to be saved",
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
