# ObjectDetectionPackage
A Python Pytorch-based package for utilities and automatization of Object Detection tasks.

# Python version compatibility
Tested python versions:
* 3.10.5

# Setup
First things first, the command to install all packages and dependencies is shown below:

`pip install requirements.txt`

## :warning: Torch version
Probably, your pytorch version may differ from the one used when developing the package due to
different cuda, cudnn, etc. versions. Therefore, it is up to the user to install it's GPU 
environment and thus not included in `requirements.txt`.

I'm aware that different pytorch versions can lead into into different dependencies, therefore,
the different torch versions used during development are listed:
* torch==1.12.1+cu113
* torchvision==0.13.1+cu113
* torchaudio==0.13.1

# How the package is meant to be used?
The command to perform a run must be:

`mlflow run . steps=X --env-manager=local`

Being X at least one of the following options (if more than one, X must be like "Option1,Option2,...,OptionN"):

* **EDA** Exploratory Data Analysis
* **PREPROCESSING** Annotation Conversion and/or Augmentations
* **MODELING** training a pytorch model

## What does each option mean and how can I configure them?
Each option has it's own MLproject file where parameters are described. Besides, a manual document 
is being developed in order to give further details on each module separatedly.

The file `config.yaml` contains the parameters each of the module requires.

## Considerations

- Latex compiler is required in order to use pylatex for report generation.
- Need of a WandB account in order to log and get artifacts.
- Currently it a pipeline is not implemented, every module must be excuted
  separatedly, so `steps` parameter can just contain one value.
- Check for log files in order to see full trace, do not be guided by
  stdout.