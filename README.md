# AFIRE: Adaptive FusionNet for Illumination-Robust Feature Extraction in Heterogeneous Imaging Environments

## Introduction

By Mingxin Yu, Xufan Miao



| Dataset | EN | MI | SF | AG | SD | VIF |
|---------|---------|---------|---------|---------|---------|---------|
|  TNO  | 7.045   | 3.867   | 11.418   | 4.419   | 43.376   | 0.814   |
|  M3FD  | 6.992   | 5.903 | 15.694 | 4.6 | 39.97 | 0.774 |
|  LLVIP | 7.273 | 4.063 | 14.016 | 4.082 | 46.863 | 1.011 |


## Installation
We utilized Python 3.7,Pytorch 1.12.0  


## Dataset

Dataset can be got from [data_illum.h5](https://github.com/linklist2/PIAFusion_pytorch)

Create datasets folder in this project, and then move the downloaded h5 file into it.

```shell
python trans_illum_data.py --h5_path 'datasets/data_illum.h5'.py
```

## Experiments
1. Train

Run [train](https://github.com/ISCLab-Bistu/DeepAdaptiveFusion/blob/main/train.py)

Type the corresponding configurations

```shell
python train.py
```

2. Test images

The testing datasets are included in [test_image](https://github.com/ISCLab-Bistu/DeepAdaptiveFusion/tree/main/test_image).

The testing code are included in [metric](https://github.com/ISCLab-Bistu/DeepAdaptiveFusion/tree/main/metric).

For `TNO` dataset, run the following code:

```shell
python metric_TNO42.py --dataset_path (your dataset path)
```

Test `M3FD` and `LLVIP` using the same method.

3. Evaluate

The evaluating code are included in [metric](https://github.com/ISCLab-Bistu/DeepAdaptiveFusion/tree/main/metric).

For `TNO`, run the corresponding python script file [metric_TNO42](https://github.com/ISCLab-Bistu/DeepAdaptiveFusion/blob/main/metric/metric_TNO42.py).

Evaluate `M3FD` and `LLVIP` using the same method.

4. Weight

Our model weight is included in [pretrained](E:\task\fusion\DAF\pretrained).


