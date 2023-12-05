# SCFusion

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
`python trans_illum_data.py --h5_path 'datasets/data_illum.h5'`

## Experiments
1. Train

Run [train](https://github.com/ISCLab-Bistu/DeepAdaptiveFusion/blob/main/train.py)

2. Test images

The testing datasets are included in [images](https://github.com/miaoxufan/SCFusion/tree/main/images).

The testing results are included in [output_images](https://github.com/miaoxufan/SCFusion/tree/main/ouput_image).

For `TNO21`,modify the `path_fusion` and `fusion_name` parameters, and then run the script file [plot_images/tno21](https://github.com/miaoxufan/SCFusion/blob/main/plot_images/tno21.py).

Test `roadscene` and `tno_vot` using the same method.

3. Evaluate

For `TNO21`, run the corresponding python script file [metric/metric_tno21](https://github.com/miaoxufan/SCFusion/blob/main/metric/metric_tno21.py).

Evaluate `roadscene` and `tno_vot` using the same method.

4. Weight

Our model weight is included in [log](https://github.com/miaoxufan/SCFusion/tree/main/log).

