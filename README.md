# Robust Balance Between Generalization and Personalization in Federated Parameter-Efficient Fine-Tuning
Here is the official implementation of the model FedTV.

![F1](FLORA.png)
## Abstract
Federated Prompt Learning (FPL) leverages the robust representation learning and remarkable generalization ability of large pre-trained Vision-Language Models (VLMS) into federated learning through prompt learning. Existing FPL methods attempt to address data heterogeneity via model personalization. However, excessive personalization leads to the compromise of model’s generalization ability, which remains a significant challenge to achieve a balance between personalization and generalization under high data heterogeneity. To address this challenge, we propose FLORA, a novel framework that combines orthogonal low-rank adaptation withattention-guided client adapter. Specifically, each client personalizes global prompt through orthogonal low-rank adaptation term, thereby achieving efficient local adaptation while maintaining the generalization of the global prompt. In addition, we introduce a lightweight attention-based adapter for the image encoder, which can enhance cross-modal alignment under nonindependent and nonidentically distributed (Non-IID) environment to further achieve the balance. Extensive experiments on multiple datasets demonstrate that our FLORA achieves superiority performance in balancing generalization and personalization over state-of-the-art methods under high data heterogeneity.

## Requirement
Follow the instruction described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install and set up necessary packages and dependencies.

## Datasets
Please follow the instructions at CoOP https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md to prepare the following datasets: Caltech101, OxfordPets, Flowers102, Food101, DTD.

For CIFAR10 and CIFAR100 datasets, please download and unzip data under DATA/ file catalog. Or simply run experiments with CIFAR10/CIFAR100 dataset, the program will download data automatically.

For DomainNet and office-caltech10 datasets, please follow the instructions of Dataset described [here](https://github.com/med-air/FedBN/blob/master/README.md).

## Training
```--root``` takes as input a path to dataset.

```--config-file```means which config file to use.

You can select variables like shots, users by changing cfg or you can change every arguments you like in scripts.

## How to run
```shell
python federated_main.py --root DATA/ --dataset-config-file configs/datasets/oxfordpets.yaml --num-users 10
```
