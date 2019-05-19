# Deeplabv3+ Implementation
This is a repo for course project of [DD2424 Deep Learning in Data Science](https://www.kth.se/social/course/DD2424/) at KTH.

This project is an Implementation of [Deeplabv3+](https://arxiv.org/abs/1802.02611) based on Pytorch. Another implementation with tensorflow:[tensorflow-deeplab-v3-plus](https://github.com/rishizek/tensorflow-deeplab-v3-plus). 

In this project we build our network based on this paper and other implementation using Pytorch. Specifically, the [modeing](https://github.com/HaoranYao/DD2424-Project/tree/master/modeling) part is based on [jfzhang95's work](https://github.com/jfzhang95/pytorch-deeplab-xception/tree/master/modeling) and we made some modifications on them.

Here is the final [report](https://github.com/HaoranYao/DD2424-Project/tree/master/result/report.pdf) for this project where you can find the details of the network and the results.


**Other Related Reference**

- the [utils](https://github.com/HaoranYao/DD2424-Project/tree/master/utils) are from existing open source python files [utils](https://github.com/jfzhang95/pytorch-deeplab-xception/tree/master/utils).

- then [main.py](https://github.com/HaoranYao/DD2424-Project/blob/master/main.py) is modified from [train.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/train.py).


## Changes from previous work

- Backbone network: ResNET101 -> ResNET156
- Datasets: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) -> [cityscapes](https://www.cityscapes-dataset.com/)

## Network Structures

- Backbone Network (ResNET156)
- ASPP
- Decoder to generate the output images

## How to run

### Requirements

- Python 3.7.0
- Pytorch 1.1.0
- tensorboardX 1.6
- CUDA 10.0

### to Run
1. Download the cityscapes data set and put them into ```/data``` folder.
2. Run 
    ```Shell
    python main.py
    ```
### Monitor the Training Process
1. Run
    ```Shell
    tensorboard --logdir log/deeplab-resnet/ &
    ```
2. Then use your browser to visit ```localhost:6006```, you will find the curves for training loss, validation loss etc.



## Conclusions

- ResNET156 does not perform better than ResNET101. The overfit may be the problem.

## Future work

- Try to solve the overfit problem, maybe use different drop-out strategy. 
- Try to modify the parameters of the ASPP module.
- Maybe create a new pipelines...




