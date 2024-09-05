# SwiftFaceFormer: An Efficient and Lightweight Hybrid Architecture for Accurate Face Recognition Applications
Luis S. Luevano, Yoanna Martínez-Díaz, Heydi Méndez-Vázquez, Miguel González-Mendoza, Davide Frey

--------

This is the code repository release for the paper "SwiftFaceFormer: An Efficient and Lightweight Hybrid Architecture for Accurate Face Recognition Applications" accepted at the 2024 27th International Conference on Pattern Recognition (ICPR).

<details>
<summary>Abstract</summary>
With the growing breakthrough of deep learning-based face recognition, the development of lightweight models that achieve high accuracy with computational and memory efficiency has become paramount, especially for deployment on embedded domains. While Vision Transformers have shown significant promising results in various computer vision tasks, their adaptability to resource-constrained devices remains a significant challenge. This paper introduces SwiftFaceFormer, a new efficient, and lightweight family of face recognition models inspired by the hybrid SwiftFormer architecture. Our proposal not only retains the representational capacity of its predecessor but also introduces efficiency improvements, enabling enhanced face recognition performance at a fraction of the computational cost. We also propose to enhance the verification performance of our original most lightweight variant by using a training paradigm based on Knowledge Distillation. Through extensive experiments on several face benchmarks, the presented SwiftFaceFormer demonstrates high levels of accuracy compared to the original SwiftFormer model, and very competitive results \hl{with respect to} state-of-the-art deep face recognition models, providing a suitable solution for real-time, on-device face recognition applications. Our code is available at https://github.com/Inria-CENATAV-Tec/SwiftFaceFormer
</details>

If you find this work useful for your research, please consider citing our work:

```BibTeX
@INPROCEEDINGS{Luevano_SwiftFaceFormer_ICPR2024,
  author={Luevano, Luis S. and Martínez-Díaz, Yoanna and Méndez-Vázquez, Heydi and Gonzalez-Mendoza, Miguel and Frey, Davide},
  booktitle={2024 27th International Conference on Pattern Recognition (ICPR)}, 
  title={SwiftFaceFormer: An Efficient and Lightweight Hybrid Architecture for Accurate Face Recognition Applications}, 
  year={2024},
  volume={},
  number={},
  pages={},
  }
```

The rest of the readme comes from the original repository

--------

## PocketNet custom

Modifications to the PocketNet repo
- Added SwiftFormer
- Added MagFace



--------

## PocketNet

## This is the official repository of the paper:
#### PocketNet: Extreme Lightweight Face Recognition Network using Neural Architecture Search and  Multi-Step Knowledge Distillation
Paper on arxiv: [arxiv](https://arxiv.org/abs/2108.10710)

Accepted at IEEE ACCESS Journal

![evaluation](https://raw.githubusercontent.com/fdbtrs/PocketNet/main/logs/tradeoff.png)


### Face recognition  model training 
Model training:
In the paper, we employ MS1MV2 as the training dataset which can be downloaded from InsightFace (MS1M-ArcFace in DataZoo)
Download [MS1MV2](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing) dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution

Unzip the dataset and place it in the data folder

Rename the config/config_xxxxxx.py to config/config.py
1. Train PocketNet with ArcFace loss
   + ./train.sh
2. Train PocketNet with template knowledge distillation
    + ./train_kd.sh
3. Train PocketNet with multi-step template knowledge distillation
    + ./train_kd.sh

| Model  | Parameters (M)| configuration | log| pretrained model| 
| ------------- | ------------- |  ------------- |------------- |------------- |
| PocketNetS-128 |0.92 |[Config](config/config_PocketNetS128.py)| [log](https://www.dropbox.com/s/hha0qp63y8w46ng/training.log?dl=0)|[Pretrained-model](https://www.dropbox.com/sh/38mhqa19xx28438/AABw64kuY4ExrE4NAQLLiJJwa?dl=0)  |
| PocketNetS-256 |0.99 |[Config](config/config_PocketNetS256.py)| [log](https://www.dropbox.com/s/tenmtzjrghaos75/training.log?dl=0)|[Pretrained-model](https://www.dropbox.com/sh/n2blqt17bg5eh1m/AAAxhWFZ2mC2hveuHzSMy0mma?dl=0) |
| PocketNetM-128 |1.68 |[Config](config/config_PocketNetM128.py) | [log](https://www.dropbox.com/s/o0vnxns6hmmj1rg/training.log?dl=0)|[Pretrained-model](https://www.dropbox.com/sh/a8qgqkyryli0nl2/AABPlP5fmiZzlN8IV64BBGica?dl=0)  |
| PocketNetM-256 |1.75 |[Config](config/config_PocketNetM256.py)| [log](https://www.dropbox.com/s/lqs47v4rc5g7425/training.log?dl=0) |[Pretrained-model](https://www.dropbox.com/sh/4dz14jgynrmsdgb/AAAsfYtKBXg1tPuK7RwzDbGva?dl=0)  |



All code has been trained and tested using  Pytorch 1.7.1

## Face recognition evaluation
##### evaluation on LFW, AgeDb-30, CPLFW, CALFW and CFP-FP: 
1. download the data from their offical webpages.
2. alternative: The evaluation datasets are available in the training dataset package as bin file
3. Rename the configuration file in config directory based on the evaluation model e.g. rename config_PocketNetM128.py to config.py to evaluate the PocketNetM128
4. set the config.rec to dataset folder e.g. data/faces_emore
5. set the config.val_targets for list of the evaluation dataset
6. download the pretrained model from link the previous table
7. set the config.output to path to pretrained model weights
8. run eval/eval.py
9. the output is test.log contains the evaluation results over all epochs

##### evaluation on IJB-B and IJB-C: 

1. Please apply for permissions from NIST before your usage [NIST_Request](https://nigos.nist.gov/datasets/ijbc/request)
2. run eval/IJB/runIJBEval.sh
 

### Differentiable architecture search training
The code of NAS is available under NAS
### To-do 
- [x] Add pretrained model
- [x] Training configuration
- [x] Add NAS code
- [x] Add evaluation results

 
 
If you use any of the provided code in this repository, please cite the following paper:
```
@article{boutros2021pocketnet,
  author    = {Fadi Boutros and
               Patrick Siebke and
               Marcel Klemt and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {PocketNet: Extreme Lightweight Face Recognition Network Using Neural
               Architecture Search and Multistep Knowledge Distillation},
  journal   = {{IEEE} Access},
  volume    = {10},
  pages     = {46823--46833},
  year      = {2022},
  url       = {https://doi.org/10.1109/ACCESS.2022.3170561},
  doi       = {10.1109/ACCESS.2022.3170561},
}
```


## License

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```
