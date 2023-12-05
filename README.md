# SCFusion

## Introduction

By Mingxin Yu, Xufan Miao



| Dataset | EN | MI | SF | SD | VIF | MS_SSIM |
|---------|---------|---------|---------|---------|---------|---------|
|  TNO21  | 7.002   | 3.944   | 9.993   | 47.101   | 0.955   | 0.936   |
|  RoadScene  | 7.272   | 3.776 | 10.302 | 48.074 | 0.905 | 1.075 |
|  Tno_Vot | 6.785 | 5.226 | 7.046 | 42.334 | 0.971 | 1.005 |


## Installation
We utilized Python 3.7,Pytorch 1.12.0  


## Dataset

Dataset can be got from [data_illum.h5](https://github.com/linklist2/PIAFusion_pytorch)


## Experiments
1. Train

Run [train_fusionnet](https://github.com/miaoxufan/SCFusion/blob/main/train_fusionnet.py)

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

