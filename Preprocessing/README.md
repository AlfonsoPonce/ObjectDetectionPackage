# Preprocessing Module
## Main goal :goal_net:
This module performs some Preprocessing steps. Currently are included:

- Annotation Conversion.
- Synthetic Augmentations through albumentations.

## What parameters are involved? 
Parameters and their descriptions are shown in MLproject.

## What does each individual submodule/script/function/etc? :frowning_person:
Due to the fact that it would be so extensive to be written in a README,
each submodule/script/function/etc. purpose has been documented in docstrings.

As soon as possible, an in-depth document will be released in order to clarify
all doubts. The main goal of this README is to clarify the module's usage.

# :warning: IMPORTANT CONSIDERATION :warning:
Images, labels and augmentations file path must be written in absolute paths. 

# About label conversion :curly_loop:
## What kinds of label conversions are allowed? :thinking:
Currently, the following conversions are allowed:

- :white_check_mark: VOC :arrows_counterclockwise: YOLO
- :white_check_mark: VOC :arrows_counterclockwise: COCO

# About Augmentations
## How albumentations is integrated?
Currently, it is only possible to perform one type of augmentation at
a time, that means, it is not possible to execute a pipeline of
augmentations.

To introduce a new albumentations's augmentation, refer to the `transformations\`
folder as it contains a bunch of examples.

## Where are transformed images and labels stored?
Currently, they are stored in a folder inside provided image and label
folders. The name of both new created folders is the same as the augmentation
applied.



