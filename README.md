# Algorithmic encoding of protected characteristics and its implications on performance disparities

![Components of a deep neural networks](assets/network.png "Components of a deep neural networks")

This repository contains the code for the paper
> B. Glocker, C. Jones, M. Bernhardt, S. Winzeck. [_Algorithmic encoding of protected characteristics and its implications on performance disparities_](https://arxiv.org/abs/2110.14755). 2021. pre-print arXiv:2110.14755. _under review_

## Dataset

The CheXpert imaging dataset together with the patient demographic information used in this work can be downloaded from https://stanfordmlgroup.github.io/competitions/chexpert/.

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n chexploration python=3
   conda activate chexploration
   ```
   
Install PyTorch using conda:
   
   ```shell
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   ```
   
### Setup Python environment using virtualenv

Create and activate a Python 3 virtual environment:

   ```shell
   virtualenv -p python3 <path_to_envs>/chexploration
   source <path_to_envs>/chexploration/bin/activate
   ```
   
Install PyTorch using pip:
   
   ```shell
   pip install torch torchvision
   ```
   
### Install additional Python packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn pytorch-lightning scikit-learn scikit-image tensorboard tqdm openpyxl
   ```

### Requirements

The code has been tested on Windows 10 and Ubuntu 18.04/20.04 operating systems. The data analysis does not require any specific hardware and can be run on standard laptop computers. The training and testing of the disease detection models requires a high-end GPU workstation. For our experiments, we used a NVIDIA Titan X RTX 24 GB.

### How to use

In order to replicate the results presented in the paper, please follow these steps:

1. Download the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), copy the file `train.csv` to the `datafiles` folder
2. Download the [CheXpert demographics data](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf), copy the file `CHEXPERT DEMO.xlsx` to the `datafiles` folder
3. Run the notebook [`chexpert.sample.ipynb`](notebooks/chexpert.sample.ipynb) to generate the study data
4. Adjust the variable `img_data_dir` to point to the imaging data and run the following scripts
   - Run the script [`chexpert.disease.py`](prediction/chexpert.disease.py) to train a disease detection model
   - Run the script [`chexpert.sex.py`](prediction/chexpert.sex.py) to train a sex classification model
   - Run the script [`chexpert.race.py`](prediction/chexpert.race.py) to train a race classification model
5. Run the notebook [`chexpert.predictions.ipynb`](notebooks/chexpert.predictions.ipynb) to evaluate all three prediction models
6. Run the notebook [`chexpert.explorer.ipynb`](notebooks/chexpert.explorer.ipynb) for the unsupervised exploration of feature representations

Additionally, there are scripts [`chexpert.sex.split.py`](prediction/chexpert.sex.split.py) and [`chexpert.race.split.py`](prediction/chexpert.race.split.py) to run SPLIT on the disease detection model. The default setting in all scripts is to train a DenseNet-121 using the training data from all patients. The results for models trained on subgroups only can be produced by changing the path to the datafiles (e.g., using `full_sample_train_white.csv` and `full_sample_val_white.csv` instead of `full_sample_train.csv` and `full_sample_val.csv`).

Note, the Python scripts also contain code for running the experiments using a ResNet-34 backbone which requires less GPU memory.

### Trained models

All trained models, feature embeddings and output predictions can be found [here](https://imperialcollegelondon.box.com/s/bq87wkuzy14ctsyf8w3hcikwzu8386jj). These can be used to directly reproduce the results presented in our paper using the notebooks [`chexpert.predictions.ipynb`](notebooks/chexpert.predictions.ipynb) and [`chexpert.explorer.ipynb`](notebooks/chexpert.explorer.ipynb).

### Expected outputs and runtimes

The scripts for disease detection, sex classification, and race classification will produce outputs in csv format that can be processed by the evaluation code in the Jupyer notebooks. The notebooks will produce figures and plots either in png or pdf format.

Training the models for disease detection, sex classification, and race classification will take about three hours each on a high-end GPU workstation. Running the data analysis code provided in the notebooks takes several minutes on a standard laptop computer.
   
## Funding sources
This work is supported through funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 757173, [Project MIRA](https://www.project-mira.eu), ERC-2017-STG) and by the [UKRI London Medical Imaging & Artificial Intelligence Centre for Value Based Healthcare](https://www.aicentre.co.uk/).

## License
This project is licensed under the [Apache License 2.0](LICENSE).
