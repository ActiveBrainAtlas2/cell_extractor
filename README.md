<a name="readme-top"></a>

<div align="center">
 
<h3 align="center">Marked Cell Detection</h3>

  <p align="center">
    <a href="https://github.com/ActiveBrainAtlas2/cell_extractor">View Demo</a>
    ·
    <a href="https://github.com/ActiveBrainAtlas2/cell_extractor/issues">Report Bug</a>
    ·
    <a href="https://github.com/ActiveBrainAtlas2/cell_extractor/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project is meant to develop a detector for marked cells in brain images based on cell shape features. The functions are:
* Extracting cells from brain images
* Evaluating cells to be negative, sure positive and unsure positive
* Testing detection accuracy via a simplified annotation process
* Estimating cell density in brain regions

Developed with Python 3.10.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

To get a local copy up and running, follow these simple example steps.

### Prerequisites

1. Please ensure you have any tool to create virtual environments. Our environment setting uses `virtualenv`.
* virtualenv
  ```bash
  sudo apt install virtualenv
  ```
2. `ImageMagick` is required to convert images. The installment guide is [here](https://imagemagick.org/script/download.php).

### Installation

1. Clone the repo to your own computer.
   ```bash
   git clone https://github.com/ActiveBrainAtlas2/cell_extractor.git
   ```
2. Configure environment variables in `configure_env.sh`
    ```bash
    # Directory to create virtual environment
    export venv='CHANGE TO YOUR DIRECTORY' 
    ```
    Note: Your directory should be out of the project directory to avoid uploading them to Github.

3. When the configuration file is set, run the following command to activate the environment variables and the virtual environment. Essential packages listed in `requirements.txt` would be installed meanwhile.
    ```bash
    source variables_env.sh
    source configure_env.sh
    ```
You can also set up the virtual environment by your familiar way. Install the essential packages by running the following command:
```bash
pip install -r $PROJECT_DIR/requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Running a marked cell detector involves the following stages:

1. full aligned brain images> 2. tiff image tiles> 3. cell examples> 4. cell features> 5.detection result

### Step 1. Full aligned images

​    The full resolution, within stack aligned images are used for the cell detection. Fluorescence images and nissel stain images are processed together in our project.

There are some example images in [data_for_test](https://github.com/ActiveBrainAtlas2/cell_extractor/blob/main/data_for_test).


### Step 2. Generate image tiles

The full resolution images are too big to work with, therefore we break them down to smaller chunks for processing with the script [generate_tif_tiles.py](https://github.com/ActiveBrainAtlas2/cell_extractor/blob/main/cell_extractor/scripts/generate_tif_tiles.py).

```bash
python $SCRIPT_DIR/generate_tif_tiles.py --animal XXX --fluorescence_image data_for_test/fluorescence_image --nissel_stain_image data_for_test/nissel_stain_image --disk output_directory
```
```
optional arguments:
  -h, --help            show this help message and exit
  --animal ANIMAL       Animal ID
  --fluorescence_image FLUORESCENCE_IMAGE
                        Where fluorescence images are stored
  --nissel_stain_image NISSEL_STAIN_IMAGE
                        Where nissel stained images are stored
  --disk DISK           Storage Disk
  --njobs NJOBS         Number of parallel jobs
```

### Step 3. Generate cell examples

The cell examples are patches of images from both fluorescent channel and nissel stain channel. Each cell example represent a candidate for cell detection.  The candidates are found by the following steps:

1. Blurring the image

2. Generate an image mask based an intensity threshold, all pixels with intensity > threshold will be labeled as 1 and the rest 0.

3. Finding the connected segments of the images mask with CV2

4. All connected segments with area <100000 are considered as cell candidates

5. Candidates on the edges of the tiles are excluded

Generate examples by running the following:
```bash
python $SCRIPT_DIR/parallel_create_examples.py --animal XXX --disk output_directory --njobs 7
```
```
optional arguments:
  -h, --help       show this help message and exit
  --animal ANIMAL  Animal ID
  --disk DISK      storage disk
  --njobs NJOBS    Number of parallel jobs
```

### Step 4. Calculate Cell Features

A set of image features are calculated from each cell candidate.  The features are used as the input to the machine learning algorithm.

```bash
python $SCRIPT_DIR/parallel_create_features.py --animal XXX --disk output_directory --njobs 7
```
```
optional arguments:
  -h, --help       show this help message and exit
  --animal ANIMAL  Animal ID
  --disk DISK      storage disk
  --njobs NJOBS    Number of parallel jobs
```

### Step 5. Detect Cells

In this step, 30 previously trained models are used to calculate a prediction score for features calculated in step 4.  The mean and standard deviation of the 30 detectors are then used to make a decision if a candidate is a sure or unsure detection.

```bash
python $SCRIPT_DIR/detect_cell_for_one_brain.py --animal XXX --disk output_directory --round 1
```
```
optional arguments:
  -h, --help       show this help message and exit
  --animal ANIMAL  Animal ID
  --disk DISK      storage disk
  --round ROUND    model version
```
We are trying to improve the models with new marked cells round by round. The argument 'ROUND' refers to the model version of each round.

### Examining the result

You can get the detection result by using the function `load_detections` in the class [CellDetectorBase](https://github.com/ActiveBrainAtlas2/cell_extractor/blob/main/cell_extractor/CellDetectorBase.py).

This will return a dataframe with the coordinate of the cells and the predictions. There are different places in the code refer to the coordinates in different ways. Remember that columns always comes before row, and x before y.

The detection result is stored in the predictions column in the dataFrame.  the result is -2 if it is not a cell, 0 if it is unsure and 2 if it is sure.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>