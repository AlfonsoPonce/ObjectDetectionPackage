# Module description
This module performs:
 - Preprocessing steps
 - Image augmentations

# IMPORTANT CONSIDERATION
Data must stand in _Data/_ folder. Each subfolder below _Data/_ describes a unique task.
Below _Data/task_X/_ data will be versioned. For example, initial dataset directory tree could be:
_Data/task_X/raw_data_. Inside the last directory, data will be splitted into
_images/_ and _labels_ [NO TRAIN/VAL/DATA SPLITTING].

This job is not done by the module.
