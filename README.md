# fashion-MNIST
computer vision course assignment

## version 1
### Model:
input   
  |    
conv1 (5x5x32)   
  |    
conv2 (5x5x64)   
  |   
conv3 (5x5x128)   
  |    
fc1 (128*2*2, 120)      
  |    
fc2 (120, 84)    
  |    
fc3 (84, 10)    
  |    
Cross Entropy   
### optimizer things
learning rate: 0.001   
activation: ReLU   
Dropout, BatchNorm
Augment: random horizontal flip
### Accuracy on test set
93.77%
## version 2
### Model
Ensemble 10 model in version 1 by incremental their output value
### optimizer things
learning rate: 0.001
activation: ReLU
Dropout, BatchNorm
Augment: random horizontal flip
### Accuracy on test set
94.27915335463258 %