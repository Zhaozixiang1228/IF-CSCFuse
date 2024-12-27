
## Requirements
- pytorch
- kornia
- tensorboardX
- h5py
- xlwt

## Train & Test
### Retrain and Test CSC-MMFN
The train and test codes are available lines 7-105 and 111-142 of `train.py`. If you want to retrain this network, you should:
- Please download and unzip the [dataset](https://mega.nz/folder/LQwVhZ4J#PNGzSnjkrqjPD4M7Td2jMA) into the folder `MMF_data/scale2`. My folder is organized as follows:
```
    mypath
    ├── train
    │   ├── balloons.mat 
    │   ├── beads.mat
    │   └── ...
    ├── test
    │   ├── real_and_fake_apples.mat
    │   ├── real_and_fake_peppers.mat
    │   └── ...
    ├── validation
    │   ├── paints.mat
    │   ├── photo_and_face.mat
    │   └── ...
    └── ...
```

- Run lines 7-105 for training.
- Run lines 111-142 for testing.

### Test MEFN with Pretrained Weights
A pretrained weight file is provided. If you do not want to retrain this model, please run `test.py`.

## Reference
```
@inproceedings{DBLP:conf/cvpr/ZhaoZBWCDSZLX23,
  author       = {Zixiang Zhao and
                  Jiang{-}She Zhang and
                  Haowen Bai and
                  Yicheng Wang and
                  Yukun Cui and
                  Lilun Deng and
                  Kai Sun and
                  Chunxia Zhang and
                  Junmin Liu and
                  Shuang Xu},
  title        = {Deep Convolutional Sparse Coding Networks for Interpretable Image
                  Fusion},
  booktitle    = {{CVPR} Workshops},
  pages        = {2369--2377},
  publisher    = {{IEEE}},
  year         = {2023}
}
```
