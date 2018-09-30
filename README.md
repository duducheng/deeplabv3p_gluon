# deeplabv3p_gluon
DeepLab v3+ in MXNet Gluon

# Note
* [`train_multi_gpu.py`](train_multi_gpu.py): multi-gpu training on Pascal VOC dataset, with validation.
* [`train.py`](train.py): single-gpu training on Pascal VOC dataset, with validation.
* [`evaluate.py`](evaluate.py): single-gpu evaluation on Pascal VOC validation. 
* [`extract_weights.py`](extract_weights.py): convert the weights from official model [release](https://github.com/tensorflow/models/blob/57eb3e77319ebce918b770801e0a5a4e3639593c/research/deeplab/g3doc/model_zoo.md).
* [`mylib`](mylib/): lib-style clean code.
* [`workspace`](workspace/): the notebooks where I did experiments, with messy staffs (ignore them).
* GPU version only, but it should be modified easily into a CPU version.
* My running environments, not tested with other environments:
    * Python==3.6
    * MXNet>=1.2.0 (MXNet==1.3.0 for multi-gpu `SyncBatchNorm`)
    * gluoncv==0.3.0
    * TensorFlow==1.4.0, Keras==2.1.5 (for converting the weights)
* Download the dataset
```bash
git clone https://github.com/dmlc/gluon-cv
cd gluon-cv/scripts/datasets
python pascal_voc.py
```
    
# Models
My porting on Pascal VOC validation:

|Model| EvalOS (w/ or w/o inference tricks) | mIoU (%) |
|:---:|:------:|:------:|
|[xception_coco_voc_trainaug (TF release)](https://github.com/tensorflow/models/blob/57eb3e77319ebce918b770801e0a5a4e3639593c/research/deeplab/g3doc/model_zoo.md)| 16 (w/o) <br> 8 (w/) | 82.20 <br> 83.58|
|[xception_coco_voc_trainaug (MXNet porting)](https://drive.google.com/open?id=19zxsJ6tmPuJcEBd-P93yCEFMLc7o4dPP)| 16 (w/o) <br> 8 (w/o) |79.19<br>81.82|
|[xception_coco_voc_trainaug (MXNet finetune TrainOS=16)](https://drive.google.com/open?id=1zusHNnPgpJAapPNEFu6FVWFqDm-_6_CZ)| 16 (w/o) <br> 8 (w/o) |82.75<br>82.56|
|[xception_coco_voc_trainaug (MXNet finetune TrainOS=8)](https://drive.google.com/open?id=1EG-6OwNU0JxDj-zBhMdGji3x8dIOK9jW)| 16 (w/o) <br> 8 (w/o) |82.02<br>83.14|
|[xception_voc_trainaug](https://drive.google.com/open?id=1a4f1e_GZ3FRPVKYrgDtmmRIYmtNTywyl) <br> ImageNet pretrained only ,without MSCOCO pretraining | 16 (w/o) <br> 8 (w/o) |77.06<br>76.44|

# AWS Runtime & Cost
Measured with fixing batch stats (`use_global_stats=True`), just for reference.

|Instance|GPUs|Pricing|Train OS|Speed|Train on train_aug|Eval on val|Time per epoch|Cost per epoch|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|p2.8xlarge|K80x8|7.20$/h|16<br>8|1.5s/b16<br>3.4s/b16|17.0min<br>37.5min|3.5min<br>10min<br>(BUGS: gpus do not use sufficiently during eval)|20.5min<br>47.5min|$2.5<br>$5.7|
|p3.8xlarge|V100x4|12.24$/h|16<br>8|0.5s/b16<br>3.0s/b12|5.5min<br>44.5min|0.7min<br>1.3min|6.2min<br>45.8min|$1.3<br>$9.3|

# Memo
* [x] transfer all the weights
* [x] add OS=8
* [x] test iou on PASCAL val
* [x] add training scripts
* [x] add multi-gpu training scripts
* [ ] train more and open source the best models
* [ ] ~~VOCAug dataset pull request~~
* [ ] ~~Model pull request~~
* [ ] ~~Finish pull request to gluoncv~~

# Acknowledge
This repository is a part of *MXNet summer code* hosted by AWS, TuSimple and Jiangmen. 
Specifically, I would like to thank [Hang Zhang](https://github.com/zhanghang1989) (@AWS) and Hengchen Dai (@TuSimple) for 
kind suggestions on tuning and implementation. Plus, I would like to thank AWS for 
providing generous credits for tuning the computationally intensive models. 
