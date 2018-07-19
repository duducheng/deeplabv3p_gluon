# deeplabv3p_gluon
DeepLab v3+ in MXNet Gluon (WIP)

# Note
* [`workspace`](workspace/): the notebooks where I did experiments, with messy staffs.
* [`mylib`](mylib/): the well organized "lib-style" code.
* GPU version only, but it should be modified easily into a CPU version.
* My running environments, not tested with other environments:
    * Python==3.6
    * TensorFlow==1.4.0
    * Keras==2.1.5
    * MXNet==1.2.0 (MXNet master version for multi-gpu `SyncBatchNorm`)
    
# Models
Official model release from [here](https://github.com/tensorflow/models/blob/57eb3e77319ebce918b770801e0a5a4e3639593c/research/deeplab/g3doc/model_zoo.md).

My porting on Pascal VOC validation:

|Model| EvalOS (w/ or w/o inference tricks) | mIoU (%) |
|:---:|:------:|:------:|
|xception_coco_voc_trainaug (TF release)| 16 (w/o) <br> 8 (w/) | 82.20 <br> 83.58|
|[xception_coco_voc_trainaug (MXNet porting)](https://drive.google.com/open?id=19zxsJ6tmPuJcEBd-P93yCEFMLc7o4dPP)| 16 (w/o) <br> 8 (w/o) | 79.19 <br> 81.85|
|xception_coco_voc_trainaug (MXNet finetune)| 16 (w/o) <br> 8 (w/o) | ? <br> ?|
|xception_voc_trainaug | 16 (w/o) <br> 8 (w/o) | ? <br> ?|


# Memo
* [x] transfer all the weights
* [x] add OS=8
* [x] test iou on PASCAL val
* [x] add training scripts
* [ ] add multi-gpu training scripts
* [ ] VOCAug dataset pull request
* [ ] Model pull request
* [ ] Finish pull request to gluoncv 

