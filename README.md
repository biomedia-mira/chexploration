# Assessing the inter-relationship of prediction tasks

![Components of a deep neural networks](assets/network.png "Components of a deep neural networks")

This repository will contain the code for the paper
> B. Glocker, S. Winzeck. _Assessing the inter-relationship of prediction tasks: Implications for algorithmic encoding of protected characteristics and its effect on AI performance_. 2021.

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
   
## Funding sources
This work is supported through funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 757173, [Project MIRA](https://www.project-mira.eu), ERC-2017-STG) and by the [UKRI London Medical Imaging & Artificial Intelligence Centre for Value Based Healthcare](https://www.aicentre.co.uk/).

## License
This project is licensed under the [Apache License 2.0](LICENSE).
