
## Requirements
- pytorch
- kornia
- tensorboardX
- h5py
- xlwt

## Train & Test
### Retrain and Test CSC-MMFN
The training codes are available at `Train.py`. 

### Test IVF with Pretrained Weights
A pretrained weight file is provided. If you do not want to retrain this model, please run `Test.py`.

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
