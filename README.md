# Vision-Transformer-Multiprocess-DistributedDataParallel-Apex

## Introduction
This project uses ViT to perform image classification tasks on DATA set CIFAR10. The implement of Vit and pretrained weight are from https://github.com/asyml/vision-transformer-pytorch. Different from https://github.com/Kaicheng-Yang0828/Vision-Transformer-ViT, this project use multi-process distributed training and it also use Apex to reduce GPU resource consumption.

![The architecture of ViT](https://github.com/Kaicheng-Yang0828/Vit-ImageClassification/blob/main/pic/VIT.png)

## Requirments
pytorch 1.7.1 <br>
python 3.7.3

## Install Apex
1、 git clone https://github.com/NVIDIA/apex.git <br>
2、 cd apex <br>
3、 python setup.py install

## Datasets

Download the CIFAR10 from http://www.cs.toronto.edu/~kriz/cifar.html or you can get it from https://pan.baidu.com/s/1ogAFopdVzswge2Aaru_lvw (code: k5v8), creat data floder and unzip the cifar-10-python.tar.gz under './data'

## Pre_trained model

You can download the pretrained file from https://pan.baidu.com/s/1CuUj-XIXwecxWMEcLoJzPg (code: ox9n), creat Vit_weights floder and pretrained file under ./Vit_weights 

## Train
```
python main.py 
```
## Result

Base on the pretrained weight, after one epoch, I get 98.1 Accuracy (I didn't adjust the parameters carefully, you can get better results by adjusting the parameters)

model  | dataset  | acc
---- | ----- | ------  
ViT-B_16  | CIFAR10 | 98.1 

## Attention 
1、Multi-process parallel training reduces the training time by one-fifth <br>
2、Apex reduce about 30% GPU resources under the premise of ensuring the same accuracy rate
