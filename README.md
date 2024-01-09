# MEPNet: A Model-Driven Equivariant Proximal Network for Joint Sparse-View Reconstruction and Metal Artifact Reduction in CT Images (MICCAI 2023)

[Hong Wang](https://hongwang01.github.io/), Minghao Zhou, Dong Wei, Yuexiang Li, [Yefeng Zheng](https://sites.google.com/site/yefengzheng/)

[[PDF]](https://drive.google.com/file/d/1aSkXJtmEsttFroe9DaVxMo5CoSsTImi8/view?usp=drive_link) [[Supplementary Material]](https://drive.google.com/file/d/1CBB4wfd9KHfdpBraTgVxOXdeK3yUwDPP/view?usp=drive_link)
## Abstract
Sparse-view computed tomography (CT) has been adopted as an important technique for speeding up data acquisition and decreasing radiation dose. However, due to the lack of sufficient projection data, the reconstructed CT images often present severe artifacts, which will be further amplified when patients carry metallic implants. For this joint sparse-view reconstruction and metal artifact reduction task, most of the existing methods are generally confronted with two main limitations: 1) They are almost built based on common network modules without fully embedding the physical imaging geometry constraint of this specific task into the dual-domain learning; 2) Some important prior knowledge is not deeply explored and sufficiently utilized. Against these issues, we specifically construct a dual-domain reconstruction model and propose a model-driven equivariant proximal network, called MEPNet. The main characteristics of MEPNet are: 1) It is optimization-inspired and has a clear working mechanism; 2) The involved proximal operator is modeled via a rotation equivariant convolutional neural network, which finely represents the inherent rotational prior underlying the CT scanning that the same organ can be imaged at different angles. Extensive experiments conducted on several datasets comprehensively substantiate that compared with the conventional convolution-based proximal network, such a rotation equivariance mechanism enables our proposed method to achieve better reconstruction performance with fewer network parameters.

## Requirements
Refer to **requirements.txt**. 
The following project links are needed for installing ODL and astra:

ODL: https://github.com/odlgroup/odl

Astra: https://github.com/astra-toolbox/astra-toolbox


This repository is tested under the following system settings:

Python 3.6

Pytorch 1.4.0

CUDA 10.1

GPU NVIDIA Tesla V100-SMX2

## Dataset
Please download from [SynDeepLesion](https://github.com/hongwang01/SynDeepLesion), and put the dataset into the folder "data" as:

```
data/train/train_640geo
data/train/train_640geo_dir.txt
data/train/trainmask.npy
data/test/test_640geo
data/test/test_640geo_dir.txt
data/test/testmask.npy
```

## Training
1. x8 downsampling
```
CUDA_VISIBLE_DEVICES=0  python train.py  --data_path "data/train/" --log_dir "logs/V80" --model_dir "pretrained_model/V80" --train_proj 80 
```

2. x4 downsampling
```
CUDA_VISIBLE_DEVICES=0  python train.py  --data_path "data/train/" --log_dir "logs/V160" --model_dir "pretrained_model/V160" --train_proj 160 
```

3. x2 downsampling
```
CUDA_VISIBLE_DEVICES=0  python train.py  --data_path "data/train/" --log_dir "logs/V320" --model_dir "pretrained_model/V320" --train_proj 320 
```

## Pretrained_Models
Please download from [[Google Drive]](https://drive.google.com/drive/folders/1YJi8Oh8ahT1CGGjg-cyLBIzOGyFyicuT?usp=drive_link) and put them into the folder "MEPNet/pretained_model"

## Testing
1. x8 downsampling
```
CUDA_VISIBLE_DEVICES=0  python test.py  --data_path "data/test/" --model_dir "pretrained_model/V80/MEPNet_latest.pt" --save_path "save_results/" --test_proj 80 
```

2. x4 downsampling
```
CUDA_VISIBLE_DEVICES=0  python test.py  --data_path "data/test/" --model_dir "pretrained_model/V160/MEPNet_latest.pt" --save_path "save_results/" --test_proj 160 
```

3. x2 downsampling
```
CUDA_VISIBLE_DEVICES=0  python test.py  --data_path "data/test/" --model_dir "pretrained_model/V320/MEPNet_latest.pt" --save_path "save_results/" --test_proj 320 
```

Reconstruction results can be downloaded from [[NetDisk]]()(pwd:mep)

## Experiments

<div  align="center"><img src="Figures/deeplesion.png" height="100%" width="100%" alt=""/></div>



## Performance Metric
Please refer to "metric/statistic.m" in [OSCNet](https://github.com/hongwang01/OSCNet)


## Citations
If helpful for your research, please cite our work:

```
@article{wang2023mepnet,
  title={MEPNet: A Model-Driven Equivariant Proximal Network for Joint Sparse-View Reconstruction and Metal Artifact Reduction in CT Images},
  author={Wang, Hong and Zhou, Minghao and Wei, Dong and Li, Yuexiang and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2306.14274},
  year={2023}
}
```


## License
The resources of this project are for academic use only, not for any commercial purposes. Please contact us if necessary. 

## To Do List
The NetDisk link for downloading reconstruction results.

## Contact
If you have any questions, please feel free to contact Hong Wang (Email: hongwang9209@hotmail.com)
