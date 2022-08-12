# Nucleus and Cell Segmentation from ISL Pretraining

Code from paper 'Learning with minimal effort: leveraging in silico labeling for cell and nucleus segmentation', Bioimaging Computing ECCV 2022 (https://www.bioimagecomputing.com/).

## Conda environment

environment.yml provides the needed packages to run the code. One can create associated environment using the following command line: 
```
conda env create -f environment.yml
```

## Documentation

Each model folder (dapi, cy5, segmentation_nucleus/basic, segmentation_nucleus/transfer, segmentation_cell/basic, segmentation_cell/transfer) contain, among others:
- model_params.py where all needed parameters can be set
- train.py to train model
- test.py to test model

dapi, segmentation_nucleus/basic and segmentation_nucleus/transfer (respectively cy5, segmentation_cell/basic and segmentation_cell/transfer) are supposed to be used with images from DataSet1 (respectively DataSet2) defined in data_set_managment/data_sets.py. One can change the file names there to adapt to its own data set.
