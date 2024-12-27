
## Requirements
- pytorch
- kornia
- tensorboardX

## Train & Test
### Retrain and Test CSC-MEFN
The train and test codes are available lines 7-93 and 96-103 of `train.py`. If you want to retrain this network, you should:
- Please download and unzip the [dataset](https://mega.nz/file/2MQRkI6A#UhseyXpWfe0x6jnzbSZwcIo6vd1QpJqo3S-tqUHfAAs) into the folder `MEF_data`. My folder is organized as follows:
```
    mypath
    ├── train
    │   ├── 1
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    │   ├── 2
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    ├── tvalidation
    │   ├── 1
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    │   ├── 2
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    ├── test
    │   ├── 1
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    │   ├── 2
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
```

- Run lines 7-93 for training.
- Run lines 96-103 for testing.

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
