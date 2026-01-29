# Robust Balance Between Generalization and Personalization in Federated Parameter-Efficient Fine-Tuning
Here is the official implementation of the model FedTV.

![F1](FLORA.png)
## Abstract
Federated Parameter-Efficient Fine-Tuning (FedPEFT) adapts transferable representations and remarkable generalization of Vision-Language Models (VLMs) into federated learning. Existing methods mainly address data heterogeneity via model personalization. However, current personalized methods suffer from three limitations. Excessive personalization often degrades model's generalization, fixed visual features fail to capture diverse visual cues, and conventional aggregation strategy is vulnerable to biased client drift. These limitations introduce a significant challenge to achieve a robust balance between generalization and personalization under data heterogeneity. To address this challenge, we propose \textbf{Fed}erated Parameter-Efficient Fine-Tuning with \textbf{T}extual Low Rank Adaptation and Dynamic \textbf{V}isual Fusion (FedTV), together with a simple yet effective Similarity-Aware Aggregation (SAA) strategy. Specifically, on the textual side, we introduce an orthogonal low-rank adaptation for the global prompt to construct personalized prompts, combined with orthogonal and contrastive losses, thereby achieving efficient local adaptation while maintaining the generalization of the global prompt. On the visual side, we propose Dynamic Visual Fusion to adaptively capture and balance diverse visual features, enhancing cross-modal alignment. Furthermore, SAA re-weights client shared updates based on their consistency with historical global parameters, improving aggregation stability under data heterogeneity under data heterogeneity. Extensive experiments on multiple datasets demonstrate that our FedTV achieves superior performance in balancing generalization and personalization over state-of-the-art methods under high data heterogeneity.

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
