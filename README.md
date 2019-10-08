# fashion-MNIST
computer vision course assignment

## version 1
### Model:
```
input
  |
  v
conv1 (5x5x32)
  |
  v
max_pooling (2x2)
  |
  v
conv2 (5x5x64)
  |
  v
max_pooling (2x2)
  |
  v
conv3 (5x5x128)
  |
  v
max_pooling (2x2)
  |
  v
fc1 (128*2*2, 120)
  |
  v
fc2 (120, 84)
  |
  v
fc3 (84, 10)
  |
  v
Cross Entropy
```
### optimizer things
+ learning rate: 0.001
+ activation: ReLU
+ Dropout, BatchNorm
+ Augment: random horizontal flip
### Accuracy on test set
93.76996805111821 %
## version 2
### Model
Ensemble 10 model in version 1 by incremental their output value
### optimizer things
+ learning rate: 0.001
+ activation: ReLU
+ Dropout, BatchNorm
+ Augment: random horizontal flip
### Accuracy on test set
94.27915335463258 %
## version 3
### Model
Wide-ResNet
```
input
  |
  v
conv1
  |
  v
resnet block1
  |
  v
resnet block2
  |
  v
resnet block3
  |
  v
 fc
  |
  v
Cross Entropy
```
### optimizer things
+ learning rate: 0.001
+ activation: ReLU
+ Dropout, BatchNorm
+ Augment: random horizontal flip
### Accuracy on test set
94.68849840255591 %

## Usage
#### Requirements
+ Python 3
+ CUDA (optional)
#### Download repository
```
git clone https://github.com/trqminh/fashion-MNIST.git
cd fashion-MNIST
```
#### Install libraries:
```
pip3 install -r requirements.txt
```
#### Install pytorch-cpu:
```
pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
#### or install pytorch with cuda (if you have already installed CUDA)
```
pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
#### create data/ directory in repository 's root directory, put the csv files in it (download from [here](https://www.kaggle.com/zalando-research/fashionmnist))
#### create trained_models/ directory in repository 's root directory and put the .pth files in it (download from [here](https://drive.google.com/open?id=1YE-am-pfQTdSPncHdbp66DBm2zyauw76))
#### Directory structure
```
|-- data
|   |-- fashion-mnist_test.csv
|   |-- fashion-mnist_train.csv
|-- models
|   |-- __init__.py
|   |-- my_model.py
|   |-- wide_resnet.py
|-- trained_models
|   |-- version1_model.pth
|   |-- version2_model.pth
|   |-- version3_model.pth
|-- utils
|   |-- __init__.py
|   |-- custom_data.py
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- test.py
|-- train.py
```
#### Training
+ Train the model in each version, with number of epochs (Recommend install CUDA)
```
python3 train.py --version $version --epoch $epoch
```
+ Example:
```
python3 train.py --version 3 --epoch 10
```
#### Evaluate
+ Test my trained models in each version
```
python3 test.py --version $version
```
+ Example:
```
python3 test.py --version 3
Evaluating...
Accuracy of the network on 10000 test images: 94.68849840255591 %
```
