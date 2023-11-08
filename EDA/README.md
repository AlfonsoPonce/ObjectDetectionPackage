# Exploratory Data Analysis Module
## Main goal :goal_net:
This module performs Exploratory Data Analysis in order to generate a pdf-latex-styled 
report.

This report will be useful for establishing some preprocessing steps. 

## What parameters are involved? 
Parameters and their descriptions are shown in MLproject.

## What does each individual submodule/script/function/etc? :frowning_person:
Due to the fact that it would be so extensive to be written in a README,
each submodule/script/function/etc. purpose has been documented in docstrings.

As soon as possible, an in-depth document will be released in order to clarify
all doubts. The main goal of this README is to clarify the module's usage.

# :warning: IMPORTANT CONSIDERATION :warning:
Images and labels path must be written in absolute paths. 


# :warning: NEED TO EDIT :warning:
In order to custom your automatic report file, you must edit `main.py`
adding sections, images, text, etc. A default template is given but it
is not the most _aesthetic_ one.

# :warning: Checking duplicate images :warning:
There are two versions available of the `check_duplicate_image()` function.

One of them is sequential and the other is parallel. It is **highly recommended**
to call the parallel one since the sequential is very slow. 

The parallel version calls a function named `compute_kernel()` which is in charge
of doing all the functional part. 

:stop_sign: **Do not call the above function, 
it just works when it is called by the parallel version of `check_duplicate_image()`**