# Algorithmic encoding of protected characteristics

![Components of a deep neural networks](assets/network.png "Components of a deep neural networks")

This repository contains the code for the paper
> B. Glocker, S. Winzeck. _Algorithmic encoding of protected characteristics and its implications on disparities across subgroups_. 2021. Pre-print; under review.

## Dataset

The CheXpert imaging dataset together with the patient demographic information used in this work can be downloaded from https://stanfordmlgroup.github.io/competitions/chexpert/.

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n pymira python=3
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
   
### How to use

In order to replicate the results presented in the paper, please follow these steps:

1. Download the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), copy the file `train.csv` to the `datafiles` folder
2. Download the [CheXpert demographics data](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf), copy the file `CHEXPERT DEMO.xlsx` to the `datafiles` folder
3. Run the notebook `chexpert.sample.ipynb` to generate the study data
4. Run the script `chexpert.disease.py` to train a disease detection model
5. Run the script `chexpert.sex.py` to train a sex classification model
6. Run the script `chexpert.race.py` to train a race classification model
7. Run the notebook `chexpert.predictions.ipynb` to evaluate all three prediction models
8. Run the notebook `chexpert.explorer.ipynb` for the unsupervised exploration of feature representations

Additionally, there are scripts `chexpert.sex.split.py` and `chexpert.race.split.py` to run SPLIT on the disease detection model. The default setting in all scripts is to train a DenseNet-121 using the training data from all patients. The results for models trained on subgroups only can be produced by changing the path to the datafiles (e.g., using `full_sample_train_white.csv` and `full_sample_val_white.csv` instead of `full_sample_train.csv` and `full_sample_val.csv`).

Note, the Python scripts also contain code for running the experiments using a ResNet-34 backbone which requires less GPU memory.
   
## Funding sources
This work is supported through funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 757173, [Project MIRA](https://www.project-mira.eu), ERC-2017-STG) and by the [UKRI London Medical Imaging & Artificial Intelligence Centre for Value Based Healthcare](https://www.aicentre.co.uk/).

## License
This project is licensed under the [Apache License 2.0](LICENSE).
