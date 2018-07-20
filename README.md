# deeplabv3p_gluon
DeepLab v3+ in MXNet Gluon

# Note
* [`train_multi_gpu.py`](train_multi_gpu.py): multi-gpu training on Pascal VOC dataset, with validation.
* [`train.py`](train.py): single-gpu training on Pascal VOC dataset, with validation.
* [`evaluate.py`](evaluate.py): single-gpu evaluation on Pascal VOC validation. 
* [`extract_weights.py`](extract_weights.py): convert the weights from official model [release](https://github.com/tensorflow/models/blob/57eb3e77319ebce918b770801e0a5a4e3639593c/research/deeplab/g3doc/model_zoo.md).
* [`workspace`](workspace/): the notebooks where I did experiments, with messy staffs (ignore them).
* GPU version only, but it should be modified easily into a CPU version.
* My running environments, not tested with other environments:
    * Python==3.6
    * MXNet>=1.2.0 (MXNet==1.3.0 for multi-gpu `SyncBatchNorm`)
    * TensorFlow==1.4.0, Keras==2.1.5 (for converting the weights)
    
# Models
My porting on Pascal VOC validation:

|Model| EvalOS (w/ or w/o inference tricks) | mIoU (%) |
|:---:|:------:|:------:|
|[xception_coco_voc_trainaug (TF release)]((https://github.com/tensorflow/models/blob/57eb3e77319ebce918b770801e0a5a4e3639593c/research/deeplab/g3doc/model_zoo.md))| 16 (w/o) <br> 8 (w/) | 82.20 <br> 83.58|
|[xception_coco_voc_trainaug (MXNet porting)](https://drive.google.com/open?id=19zxsJ6tmPuJcEBd-P93yCEFMLc7o4dPP)| 16 (w/o) <br> 8 (w/o) | 79.19 <br> 81.85|
|[xception_coco_voc_trainaug (MXNet finetune) (WIP)](https://drive.google.com/open?id=1zusHNnPgpJAapPNEFu6FVWFqDm-_6_CZ)| 16 (w/o) <br> 8 (w/o) | 82.75 <br> 82.56|
|xception_voc_trainaug | 16 (w/o) <br> 8 (w/o) | ? <br> ?|


# Memo
* [x] transfer all the weights
* [x] add OS=8
* [x] test iou on PASCAL val
* [x] add training scripts
* [x] add multi-gpu training scripts
* [ ] train more and open source the best models
* [ ] VOCAug dataset pull request
* [ ] Model pull request
* [ ] Finish pull request to gluoncv 

