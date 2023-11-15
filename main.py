import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from pathlib import Path
import os
import json

_steps = [
    "EDA",
    "Preprocessing",
    "Modeling",
    "Inference",
    "Deploy"
]

@hydra.main(config_name='config')
def run(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    working_directory = Path.cwd().parents[2]

    if "EDA" in active_steps:
        # Download file and load in W&B
        _ = mlflow.run(
            str(Path(hydra.utils.get_original_cwd()).joinpath("EDA")),
            "main",
            parameters={
                "images_dir": str(working_directory.joinpath(Path(config["EDA"]["DATASET_ANALYSIS"]["images_dir"]))),
                "labels_dir": str(working_directory.joinpath(config["EDA"]["DATASET_ANALYSIS"]["labels_dir"])),
                "plots_output_dir": str(working_directory.joinpath(Path(config["EDA"]["DATASET_ANALYSIS"]["plots_output_dir"]))),
                "report_save_dir": str(working_directory.joinpath(config["EDA"]["DATASET_ANALYSIS"]["report_save_dir"])),
                "artifact_name": config["EDA"]["DATASET_ANALYSIS"]["artifact_name"],
                "artifact_type": config["EDA"]["DATASET_ANALYSIS"]["artifact_type"],
                "artifact_description": config["EDA"]["DATASET_ANALYSIS"]["artifact_description"]
            },
            env_manager='local'
        )

    if "PREPROCESSING" in active_steps:
        # Download file and load in W&B
        _ = mlflow.run(
            str(Path(hydra.utils.get_original_cwd()).joinpath("Preprocessing")),
            "main",
            parameters={
                'annotation_conversion': config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["perform"],
                'class_list': config["PREPROCESSING"]["class_list"],
                'src_label_format': config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["src_label_format"],
                'dst_label_format': config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["dst_label_format"],
                'input_label_dir': str(working_directory.joinpath(config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["input_label_dir"])),
                'annotation_conversor_output_label_dir': str(working_directory.joinpath(config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["output_label_dir"])),
                'annotation_conversor_artifact_name': config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["artifact_name"],
                'annotation_conversor_artifact_type': config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["artifact_type"],
                'annotation_conversor_artifact_description': config["PREPROCESSING"]["ANNOTATION_CONVERSOR"]["artifact_description"],

                'augmentations': config["PREPROCESSING"]["AUGMENTATIONS"]["perform"],
                'image_directory': str(working_directory.joinpath(config["PREPROCESSING"]["AUGMENTATIONS"]["image_directory"])),
                'labels_directory': str(working_directory.joinpath(config["PREPROCESSING"]["AUGMENTATIONS"]["labels_directory"])),
                'augmentation_file': str(working_directory.joinpath('Preprocessing','Augmentations','transformations',config["PREPROCESSING"]["AUGMENTATIONS"]["augmentations_file"])),
                'merge_augmented_images': config["PREPROCESSING"]["AUGMENTATIONS"]["merge_augmented_images"],
                'augmentations_artifact_name': config["PREPROCESSING"]["AUGMENTATIONS"]["artifact_name"],
                'augmentations_artifact_type': config["PREPROCESSING"]["AUGMENTATIONS"]["artifact_type"],
                'augmentations_artifact_description': config["PREPROCESSING"]["AUGMENTATIONS"]["artifact_description"]
            },
            env_manager='local'
        )

    if "MODELING" in active_steps:
        print(str(OmegaConf.to_container(config["MODELING"]["train_config"])).replace(" ", ""))
        # Download file and load in W&B
        _ = mlflow.run(
            str(Path(hydra.utils.get_original_cwd()).joinpath("Modeling")),
            "main",
            parameters={
                'dataset_format': config["MODELING"]["dataset_format"],
                'class_list': config["MODELING"]["class_list"],
                'images_dir': str(
                    working_directory.joinpath(config["MODELING"]["images_dir"])),
                'labels_dir': str(
                    working_directory.joinpath(config["MODELING"]["labels_dir"])),
                'output_dir': str(
                    working_directory.joinpath(config["MODELING"]["output_dir"])),
                'batch_size': config["MODELING"]["train_config"]["batch_size"],
                'train_split': config["MODELING"]["train_split"],
                'model': config["MODELING"]["model"],
                'train_config': str(OmegaConf.to_container(config["MODELING"]["train_config"])).replace(" ", ""),
                'artifact_name': config["MODELING"]["artifact_name"],
                'artifact_type': config["MODELING"]["artifact_type"],
                'artifact_description': config["MODELING"]["artifact_description"]
            },
            env_manager='local'
        )

if __name__ == '__main__':
   run()