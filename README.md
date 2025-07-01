# **M2M-LINet**


The implementation of

[From Macro to Micro: A Lightweight Interleaved Network for Remote Sensing Image Change Detection](https://ieeexplore.ieee.org/document/10912653)

on *2025 IEEE Transactions on Geoscience and Remote Sensing*.

## **Network**


![Network Architecture](https://github.com/Sean1005-x/M2M-LINet/blob/main/temp/M2MLINet.jpg)

## **Datasets**


[**SYSU-CD**](https://github.com/liumency/SYSU-CD) is a public large-scale remote sensing image change detection (RSICD) dataset. It contains 20,000 pairs of 256×256 remote sensing images with 0.5-m resolution, covering diverse targets such as buildings, vessels, roads, and vegetation. We use its default split: 12,000/4,000/4,000 pairs for training/validation/testing.

[**WHU-CD**](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) focuses on building changes, with 1 pair of 32,507×15,354 images (0.2-m resolution). Following mainstream processing, we crop it into non-overlapping 256×256 patches and split into 6,096/762/762 for training/validation/testing.

[**LEVIR-CD+**](https://justchenhao.github.io/LEVIR/) is a large building-focused RSICD dataset, including 637 and 985 pairs of 1024×1024 images (0.5-m resolution). We follow the original split, crop into 256×256 patches, and use 10,192/5,568 pairs for training/testing.

## **Experiments**


![Experimental Results 1](https://github.com/Sean1005-x/M2M-LINet/blob/main/temp/experiment.jpg)
![Experimental Results 2](https://github.com/Sean1005-x/M2M-LINet/blob/main/temp/Visualization.jpg)


## Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{10912653,
  author={Xu, Yetong and Lei, Tao and Ning, Hailong and Lin, Shaoxiong and Liu, Tongfei and Gong, Maoguo and Nandi, Asoke K.},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={From Macro to Micro: A Lightweight Interleaved Network for Remote Sensing Image Change Detection}, 
  year={2025},
  volume={63},
  pages={1-14},
  keywords={Transformers;Feature extraction;Convolutional neural networks;Attention mechanisms;Change detection (CD);lightweight;remote sensing (RS) image},
  doi={10.1109/TGRS.2025.3548562}}
