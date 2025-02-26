# CG-GAN

This is the CG-GAN part of the official implementation of the paper: "Synergy Between Sufficient Changes and Sparse Mixing Procedure for Disentangled Representation Learning" (ICLR 2025).

## Overview

CG-GAN extends [i-StyleGAN](https://github.com/Mid-Push/i-stylegan) by introducing an additional Jacobian matrix regularization term.

## Installation

The installation follows the same requirements as [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch). Please refer to their repository for detailed environment setup.

## Dataset Preparation

To reproduce our experiments on the two-color MNIST dataset:

1. Download our processed dataset from [HuggingFace](https://huggingface.co/datasets/ShunxingFan/mnist-two-color/tree/main)
2. Place the downloaded zip file in the `CG-GAN/datasets` directory

## Training

Our training process consists of two stages:
1. Pre-training using i-StyleGAN (approximately 20,000 kimg)
2. Fine-tuning with our Jacobian regularization

### Fine-tuning Command

```bash
python train.py --outdir=training-runs \
                --data=datasets/mnist.zip \
                --gpus=4 \
                --cond_mode=flow \
                --flow_norm=1 \
                --i_dim=256 \
                --lambda_sparse=0.1 \
                --lambda_imgsparse=0.1 \
                --perturb_norm=0.1 \
                --resume=/path/to/pretrained/network-snapshot-019356.pkl
```

The training will continue for approximately 5,000 kimg until reaching a total of 25,000 kimg.

## Weights & Biases Integration

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. To use wandb:

1. Install wandb: `pip install wandb`
2. Login to wandb: `wandb login`
3. (Optional) Set your wandb entity: `export WANDB_ENTITY=your-entity-name`

If wandb is not configured, the training will continue without logging.

## Citation

This is the official implementation of:

**Synergy Between Sufficient Changes and Sparse Mixing Procedure for Disentangled Representation Learning**  
Zijian Li*, Shunxing Fan*, Yujia Zheng, Ignavier Ng, Shaoan Xie, Guangyi Chen, Xinshuai Dong, Ruichu Cai, Kun Zhang  
International Conference on Learning Representations (ICLR) 2025

If you find this work useful for your research, please cite our paper.
```bash
@inproceedings{
li2025synergy,
title={Synergy Between Sufficient Changes and Sparse Mixing Procedure for Disentangled Representation Learning},
author={Zijian Li and Shunxing Fan and Yujia Zheng and Ignavier Ng and Shaoan Xie and Guangyi Chen and Xinshuai Dong and Ruichu Cai and Kun Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=G1r2rBkUdu}
}
```
## Acknowledgments

We would like to thank the authors of [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [i-StyleGAN](https://github.com/Mid-Push/i-stylegan) for their excellent open-source implementations that served as the foundation for this work.


