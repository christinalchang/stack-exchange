# Stack Exchange: Multi-Label Tag Classification

Troy Lui, Christina Chang, Jonathan Quach

# Overview

This directory IPython Notebooks, Python scripts, and data related to the Stack Exchange Multi-Label Tag Classification Kaggle challenge. Please refer to `STA141C_ProjectReport` for more details on implementation and results.

# Contents

## Directories
* `./DataData` - Incomplete one-hot encoding tables. Please refer to the Google Drive link for the complete data.
* `./cleaned` - The cleaned text data
* `./cleaned_trim` - The cleaned text with the top-5% of tags
* `./data` - The raw text data in DataFrame format

## Notebooks 

* `BinaryRelevance.ipynb` - Used to explore BinaryRelevance Classification of posts.
* `DataFrames_Function&Full.ipynb` - Used to generate one-hot encoding of the tags in the posts per each category
* `FinalDataParameters.ipynb`- Used to find number of tags per category.
* `PowerSet.ipynb` - Used to explore PowerSet Classification of posts.
* `examine_tags.ipynb` - Used to examine the tags per post category and find the most frequent tags per category
* `keras.ipynb` - Used for the Neural Network classification of posts
* `train.ipynb` - Used to test logistic regression of the text data.
* `trim_tags_clean_text.ipynb` - Used to clean and standardize the text data.

## `.py` Scripts

* `multi-label-classification-bert.py` - Attempted code to perform BERT transfer learning
* `trim_tags_clean_text.py` - Text cleaning/standardization script

## Miscellaneous

* `model-conv1d.h5` - Neural Network data

