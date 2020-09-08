# Kaggle competition:
# [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)


Team Yuval & nosound:
====================

-   [Yuval Reina](https://www.kaggle.com/yuval6967)

-   [Zahar Chikishev](https://www.kaggle.com/zaharch)

Private Leaderboard Score: Yuval & nosound

Private Leaderboard Place: 27

The With-Context Prize

General
=======

This archive holds the code which was used to create and inference
the 27th place solution in “SIIM-ISIC Melanoma Classification” competition.

The solution consists of the following components, run consecutively

-   Prepare data and metadata

-   Training features generating neural networks

-   Training transformer neural networks based on the features and metadata

-   Ensembling

ARCHIVE CONTENTS
================

-   Main - All notebooks needed to prepare, train and inference the models 

-   exp - a python library holding files that are automatically created using some of the notebooks

Setup
=====

### HARDWARE: (The following specs were used to create the original solution)

CPU intel i9-9920, RAM 64G, GPU Tesla V100, GPU Titan RTX.


### SOFTWARE (python packages are detailed separately in requirements.txt):

OS: Ubuntu 18.04 TLS

CUDA – 10.2

Kaggle GPU docker with the following added/changed python libraries:

* Pytorch 1.6.0 (actually the nightly version - 1.6.0.dev20200528)

* Fire 

* geffnet

* sandesh 

* pretrainedmodels


DATA SETUP
==========

1.  Download train and test data from Kaggle, and the 2019 competition data from [2019 archive](https://challenge2019.isic-archive.com/data.html)

2. Update config.json file with the data directories locations:

* data - the main data location

* train_dicom -  location of train dicom files

* test_dicom -  location of test dicom files

* train_jpg -  location of train jpg files

* test_jpg -  location of test jpg files

* ISIC_2019 - location of main ISIC 2019 data

* train_ISIC_2019 - location of ISIC 2019 images

3. Create and update the name of the following folders

* train_jpg_*** -  location of processed train jpg files (replace ** with any name)

* test_jpg_*** -  location of processed test jpg files (replace ** with any name)

* features  - location for features files

* models - location for models files

* outputs - location for predictions files

Data Processing
===============

Prepare data + metadata
-----------------------

Run the prepare.ipynd notebook

Currently this notebook is set to create 400*600 images and can easily be changed.

Training Base Models 
---------------------
Run ‘basic model train_inference.ipynb’ 

The 2nd cell in this notebook contains the model type which could be changed and the seeds.

Beware Running the full notebook can take a couple of days, depending on the model size and GPU type. 


Prepare feature vectors 
--------------------------
After the models are trained you’ll need to prepare the feature vectors for training using the notebook ‘calculate features.ipynb’.

The 2nd cell holds the configuration as before.

you don’t need to prepare features for the test data, as it was already done in the previous stage, but if you didn't run the training notebook and want to prepare the test features, you can use 'calculate test_features.ipynb'

TrainingTransformer models 
--------------------------
Run ‘patient transformer model train_inference.ipynb’ 

Here there is some more names to set in the 2nd cell, follow the example

Ensemble
-------------------------- 
Use Ensemble.ipynb to do the ensembling.

Edit the 4th cell to reflect the names of the output files created while running the precious notebooks. 
