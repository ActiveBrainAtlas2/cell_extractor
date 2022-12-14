.. Marked Cell Detection documentation master file, created by
   sphinx-quickstart on Wed Dec  7 16:41:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Marked Cell Detection's documentation!
=================================================

The cell detector repository contains code for training and applying XGBoost detectors for cell detection on fluorescent images.

The cell detection process involves the following steps:
1. Aligned Brain Image > 2. Tiff Image Tiles> 3. Cell Candidates> 4. Cell Features> 5.Detection Result

The within the stack aligned images are first broken down to smaller tiles for faster processing.  Cell candidates are then generated from the image tiles.  The cell candidates are patches of images from both Nissl stain and the target channel for detection. 

The candidates are found by the following step:
1. Subtract the image background by subtracting a blurred version from the original image.
2. Generate an image mask based on an intensity threshold.  
3. Finding the connected segments of the images mask.
4. All connected segments with area <100 microns squared are considered as cell candidates
5. candidates on the edges of the tiles are excluded

A set of image features are then calculated from each cell candidate. The features are used as the input to the machine learning algorithm. During detection, 30 previously trained models are used to calculate a prediction score for the extracted features. The mean and standard deviation of the scores from 30 XGBoost detectors are then used to decide if a candidate is a sure or unsure detection.

The two modules:
* cell_extractor - most of the main modules are located in this main folder of the repository
* lib - a set of supplementary tools


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
