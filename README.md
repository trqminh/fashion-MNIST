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
93.77%
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
94.6685303514377 %


