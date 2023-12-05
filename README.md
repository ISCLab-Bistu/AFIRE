# SCFusion

## Introduction

By Xufan Miao, Mingxin Yu



| Dataset | EN | MI | SF | SD | VIF | MS_SSIM |
|---------|---------|---------|---------|---------|---------|---------|
|  TNO21  | 7.002   | 3.944   | 9.993   | 47.101   | 0.955   | 0.936   |
|  RoadScene  | 7.272   | 3.776 | 10.302 | 48.074 | 0.905 | 1.075 |
|  Tno_Vot | 6.785 | 5.226 | 7.046 | 42.334 | 0.971 | 1.005 |


## Installation
We utilized Python 3.7,Pytorch 1.12.0  


## Dataset

Download [MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) and [KAIST](https://soonminhwang.github.io/rgbt-ped-detection/)

Change the paths in the args file: [args_fusion](https://github.com/miaoxufan/SCFusion/blob/main/args_fusion.py)

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


## Acknowledgement
The code in this project is borrowed from ![RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest).

```
@article{li2021rfn,
  title={RFN-Nest: An end-to-end residual fusion network for infrared and visible images},
  author={Li, Hui and Wu, Xiao-Jun and Kittler, Josef},
  journal={Information Fusion},
  volume={73},
  pages={72--86},
  month={March},
  year={2021},
  publisher={Elsevier}
}
```