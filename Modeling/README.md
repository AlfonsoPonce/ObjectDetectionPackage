# Modeling Module
## Main goal :goal_net:
The aim of this module is to perform all modeling-related tasks such as:
- :top: Model definition and acquisition
- :top: Model training
- :top:  Model selection
- And so on

## What parameters are involved? 
Parameters and their descriptions are shown in MLproject.

## What does each individual submodule/script/function/etc? :frowning_person:
Due to the fact that it would be so extensive to be written in a README,
each submodule/script/function/etc. purpose has been documented in docstrings.

As soon as possible, an in-depth document will be released in order to clarify
all doubts. The main goal of this README is to clarify the module's usage.

# :warning: IMPORTANT CONSIDERATION :warning:
- Images and labels path must be written in absolute paths. 
- The name of the model must be the same as the filename which contains the `create_model()`
  function.
- Currently it is only supported **PASCAL VOC** datasets.

# How can I add a new model? :thinking:
1. Go to `Model_Zoo\model_repo\*` and add a filename with the name you
want to instantiate your model. 
2. In that filename you must provide a function called `create_model()`
   which returns an object of your model class.
3. Add the call to custom model to `Zoo.py`