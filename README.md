# Best training throughput optimization for four-stage STR framework
| [paper](https://arxiv.org/abs/1904.01906) |

Official PyTorch implementation of our four-stage STR framework, that most existing STR models fit into. <br>
Using this framework allows for the module-wise contributions to performance in terms of accuracy, speed, and memory demand, under one consistent set of training and evaluation datasets. <br>
Such analyses clean up the hindrance on the current comparisons to understand the performance gain of the existing modules. <br><br>
<img src="./figures/training_throughput.png" width="1000" title="training-throughput">


## Getting Started
### Dependency
- This work was tested with PyTorch 1.3.1, CUDA 10.2, python 3.6 and Ubuntu 18.04.
- requirements : lmdb, pillow, torchvision, nltk, natsort
```
pip3 install lmdb pillow torchvision nltk natsort
```
### Run demo with pretrained model
1. Download pretrained model from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
2. Add image files to test into `demo_image/`
3. Run demo.py (add `--sensitive` option if you use case-sensitive model)
```
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```

#### prediction results

| demo images | [TPS-ResNet-BiLSTM-Attn](https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9) | [TPS-ResNet-BiLSTM-Attn (case-sensitive)](https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY) |
| ---         |     ---      |          --- |
| <img src="./demo_image/demo_1.png" width="300">    |   available   |  Available   |
| <img src="./demo_image/demo_2.jpg" width="300">      |    shakeshack    |   SHARESHACK    |
| <img src="./demo_image/demo_3.png" width="300">  |   london   |  Londen   |
| <img src="./demo_image/demo_4.png" width="300">      |    greenstead    |   Greenstead    |
| <img src="./demo_image/demo_5.png" width="300" height="100">    |   toast   |  TOAST   |
| <img src="./demo_image/demo_6.png" width="300" height="100">      |    merry    |   MERRY    |
| <img src="./demo_image/demo_7.png" width="300">    |   underground   |   underground  |
| <img src="./demo_image/demo_8.jpg" width="300">      |    ronaldo    |    RONALDO   |
| <img src="./demo_image/demo_9.jpg" width="300" height="100">    |   bally   |   BALLY  |
| <img src="./demo_image/demo_10.jpg" width="300" height="100">      |    university    |   UNIVERSITY    |


## Reference
[1] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman. Synthetic data and artificial neural networks for natural scenetext  recognition. In Workshop on Deep Learning, NIPS, 2014. <br>
[2] A. Gupta, A. Vedaldi, and A. Zisserman. Synthetic data fortext localisation in natural images. In CVPR, 2016. <br>
[3] D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i Big-orda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazan, andL. P. De Las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013. <br>
[4] D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R.Chandrasekhar, S. Lu, et al. ICDAR 2015 competition on ro-bust reading. In ICDAR, pages 1156–1160, 2015. <br>
[5] A. Mishra, K. Alahari, and C. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012. <br>
[6] K. Wang, B. Babenko, and S. Belongie. End-to-end scenetext recognition. In ICCV, pages 1457–1464, 2011. <br>
[7] S. M. Lucas, A. Panaretos, L. Sosa, A. Tang, S. Wong, andR. Young. ICDAR 2003 robust reading competitions. In ICDAR, pages 682–687, 2003. <br>
[8] T. Q. Phan, P. Shivakumara, S. Tian, and C. L. Tan. Recognizing text with perspective distortion in natural scenes. In ICCV, pages 569–576, 2013. <br>
[9] A. Risnumawan, P. Shivakumara, C. S. Chan, and C. L. Tan. A robust arbitrary text detection system for natural scene images. In ESWA, volume 41, pages 8027–8048, 2014. <br>
[10] B. Shi, X. Bai, and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. In TPAMI, volume 39, pages2298–2304. 2017. 


## Contact
Feel free to contact us if there is any question: <br>
for code/paper Jeonghun Baek ku21fang@gmail.com; for collaboration hwalsuk.lee@navercorp.com (our team leader).


## License
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

