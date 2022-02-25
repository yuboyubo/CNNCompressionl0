# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 100
LR = 0.001          # learning rate
DOWNLOAD_MNIST = True# True  # False

# # Mnist 
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # transform PIL.Image or numpy.ndarray into torch.FloatTensor (C x H x W), and normalize [0.0, 1.0]
    download=DOWNLOAD_MNIST,          
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())

# batch training 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


class MLP(nn.Module):
    def __init__(self, net_dims, p=0, activation=nn.ReLU,dropout=nn.Dropout):
        """Constructor for multi-layer perceptron pytorch class
        params:
            * net_dims: list of ints  - dimensions of each layer in neural network
            * activation: func        - activation function to use in each layer
                                      - default is ReLU
        """
        super(MLP, self).__init__()
        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1], bias=True))

            # use activation function if not at end of layer
            if i != len(net_dims) - 2:
                layers.append(dropout(p))
                layers.append(activation())

        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.uniform_(-1,1)

    def forward(self, x):
        loss = self.net(x)

        return loss

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out')
              m.bias.data.uniform_(-1,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out

def test(model, test_x, test_y, device):
    model.eval()
    if 'MLP' in model.__class__.__name__:
        test_x = test_x.view(test_x.shape[0], -1)
    test_x, test_y = test_x.to(device), test_y.to(device)
    test_output = model(test_x)
    pred_y = torch.max(test_output, 1)[1].data#.numpy()
    test_acc = float((pred_y == test_y.data).sum()) / float(test_y.size(0))#.astype(int)
    return test_acc

def cal_num_nonzero(model):
    cnt = 0
    for param in model.parameters():
        cnt += param.detach().nonzero().size(0)

    return cnt

def calc_layerwise_sparsity(model):
    nnz_sum = 0
    param_sum = 0
    for i,(name,param) in enumerate(model.named_parameters()):
        nonzeros = param.detach().nonzero().size(0)
        nnz_sum += nonzeros
        param_sum += param.numel()
        if 'weight' in name:
            num_original_weights = param.numel()
            num_weights = nonzeros
        else:
            num_original_bias = param.numel()
            num_bias = nonzeros

            num_orig_param = num_original_weights + num_original_bias
            nnz = num_weights + num_bias
            if 'MLP' in model.__class__.__name__: # For display reason, we write the following code to show the names of layers correctly.
                print('FC%d'%((i+1)/2),nnz,'/', num_orig_param,'(%.2f%%)' % (nnz/num_orig_param*100))
            elif 'conv' in name:
                print('Conv%d'%((i+1)/2),nnz,'/', num_orig_param,'(%.2f%%)' % (nnz/num_orig_param*100))
            elif 'fc' in name:
                print('FC%d'%((i+1)/4),nnz,'/', num_orig_param,'(%.2f%%)' % (nnz/num_orig_param*100))
    print('Overall sparsity ratio:',nnz_sum,'/', param_sum,'(%.2f%%)' % (nnz_sum/param_sum*100))

def remainingNeurons(model):
    neurons = []
    for name,param in model.named_parameters():
        if len(param.detach().shape)==2: # to calculate the active neurons in fully connected layers
            weights = param.detach().clone()
            neuron = sum((weights!=0).any(axis=0))
            neurons.append(neuron.item())
        if 'conv' in name:
            if 'weight' in name:
                weights = param.detach().clone()
                weights = weights.view(weights.size(0),-1) # reshape weights into 2D        
            else:
                biases = param.detach().clone()
                wb = torch.cat((weights,torch.unsqueeze(biases,1)),1) #expanding rows to include biases
                neuron = sum((wb!=0).any(axis=1)) #if there is at least one value in a row, then this row(neuron) is active. 
                neurons.append(neuron.item())

    return neurons

def cal_FLOPs(model):
    # to compute the FLOPs of a model
    original_operations = 0
    pruned_operations = 0
    for name,param in model.named_parameters():
        if 'weight' in name:
          original_operations += 2*param.detach().shape[0]*param.detach().shape[1] # to get the number of operations when calculating W*X in original network
          pruned_operations += 2*param.detach().norm(0) # to get the number of operations when calculating W*X in pruned network
        else:
          original_operations += param.detach().size(0) # to get the number of operations when adding up the bias in original network
          original_operations += param.detach().size(0) # to get the number of operations when calculating Relu activations, and there are no activation operations in the input layer
          pruned_operations += param.detach().norm(0) # to get the number of operations when adding up the bias in pruned network
    original_operations -= param.detach().size(0) # although we use softmax in Pytorch's crossEntropy function, we don't consider it since there's no difference here between original network and pruned work
    pruned_operations += sum(remainingNeurons(model)[1:]) # In the pruned network, the number of operations for Relu is exactly the number of active neurons in the middle layers
    FLOPs = pruned_operations/original_operations
    return FLOPs

def FLOPs_LeNet5_mnist(model):
  '''
  Input: 1x28*28
  Conv1: 20*1*5*5(kernel) -> 20*24*24(Conv1 and Relu) -> 20*12*12(MaxPooling)
  Conv2: 50*20*5*5(kernel)-> 50* 8* 8(Conv2 and Relu) -> 50* 4* 4(MaxPooling)
  Flatten: 50* 4* 4 = 800
  Fc1:     500*800 
  Fc2:     10 *500
  '''
  original_operations = 0
  pruned_operations = 0
  feat_size = 28
  neurons = remainingNeurons(model) # e.g. if neurons is [3,4], neurons2 would be [3,3,4,4]
  neurons2 = [n for n in neurons for _ in (0,1)] # duplicate of neurons
  for (name,param),neuron in zip(model.named_parameters(),neurons2):
    params = param.detach()
    p_shape = params.shape
    if len(p_shape)==4:
      feat_size -= 5-1
      feat_p_size = feat_size / 2 # size of feature map after pooling
      # conv operations + Relu activations + maxpooling
      original_operations += p_shape[0]*(p_shape[1]*(feat_size)**2*(5*5+5*5-1+1)+feat_size**2+feat_p_size**2*(2*2-1))
      pruned_operations += feat_size**2*(2*params.norm(0)-1) + neuron*(feat_size)**2 + neuron*feat_p_size**2*(2*2-1)
    elif len(p_shape)==2:
      original_operations += 2*p_shape[0]*p_shape[1] + p_shape[0]
      pruned_operations += 2*params.norm(0)
    if len(p_shape)==1:
      pruned_operations += feat_size**2*params.norm(0)*neuron # for each filter, we need to consider bias for feat_size*feat_size on active filter
  # Relu in Fc2 and softmax in the output layer
  pruned_operations += neurons[-1] + p_shape[0]
  FLOPs = pruned_operations/original_operations
  return FLOPs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('You are using %s.\n'%device)
net_dims = [784,300,100,10]
model_LeNet300 = MLP(net_dims)
model_LeNet300 = model_LeNet300.to(device)
print('*** Testing DNN_PALM_L0 pruning algorithm on LeNet-300-100 ***')
print('----------------------------------------------------------------------------')
print('Loading a pruned LeNet-300-100 model...')
pruned_LeNet300_L0_file = 'pruned_LeNet300_L0_0.9819_nnz_3209.pkl'
model_LeNet300.load_state_dict(torch.load(pruned_LeNet300_L0_file,map_location=device))#map_location=torch.device('cpu')
print('Pruned LeNet-300-100 model has been loaded successfully.\n')


print('*** Testing the number of nonzeros of the pruned LeNet-300-100 model obtained by DNN_PALM_L0 pruning algo...')
nnz = cal_num_nonzero(model_LeNet300)
print('The pruned LeNet-300-100 has %d nonzero parameters.\n'%nnz)

print('*** Testing the test accuracy of the pruned model...')
test_acc = test(model_LeNet300, test_x, test_y, device)
print('The test accuracy of the pruned LeNet-300-100 model is %.2f%%.\n'%(100*test_acc))

print('*** Testing layerwise sparsity ratio of the pruned model...')
calc_layerwise_sparsity(model_LeNet300)


print('\n*** Testing DNN_PALM_L0_Group_Lasso pruning algorithm on LeNet-300-100 ***')
print('----------------------------------------------------------------------------')

print('Loading a pruned LeNet-300-100 model...')
pruned_LeNet300_SparseGL_file = 'pruned_LeNet300_SparseGL_0.9822_nnz_4880.pkl'
model_LeNet300.load_state_dict(torch.load(pruned_LeNet300_SparseGL_file,map_location=device))#map_location=torch.device('cpu')
print('Pruned LeNet-300-100 model has been loaded successfully.\n')

print('*** Testing the number of nonzeros of the pruned model...')
nnz = cal_num_nonzero(model_LeNet300)
print('The pruned LeNet-300-100 has %d nonzero parameters.\n'%nnz)
print('*** Testing the architecture of the pruned model...')
print('The original structure is',[i for i in net_dims[:-1]])
neurons = remainingNeurons(model_LeNet300)
print('The pruned structure is',[i for i in neurons])

print('\n*** Testing the FLOPs of the pruned model...')
FLOPs = cal_FLOPs(model_LeNet300)
print('The FLOPs of the pruned LeNet-300-100 model is %.2f%%.\n'%(100*FLOPs))

print('\n*** Testing the test accuracy of the pruned model...')
test_acc = test(model_LeNet300, test_x, test_y, device)
print('The test accuracy of the pruned LeNet-300-100 model is %.2f%%.\n'%(100*test_acc))

print('*** Testing layerwise sparsity ratio of the pruned LeNet-300-100 model obtained by DNN_PALM_L0_Group_Lasso pruning algo...')
calc_layerwise_sparsity(model_LeNet300)

model_LeNet5 = LeNet5()
model_LeNet5 = model_LeNet5.to(device)
print('\n*** Testing DNN_PALM_L0 pruning algorithm on LeNet-5 ***')
print('----------------------------------------------------------------------------')
print('Loading a pruned LeNet-5 model...')
pruned_LeNet5_L0_file = 'pruned_LeNet5_L0_0.9911_nnz_2092.pkl'
model_LeNet5.load_state_dict(torch.load(pruned_LeNet5_L0_file,map_location=device))#map_location=torch.device('cpu')
print('Pruned LeNet-5 model has been loaded successfully.\n')

print('*** Testing the number of nonzeros of the pruned LeNet-5 model obtained by DNN_PALM_L0 pruning algo...')
nnz = cal_num_nonzero(model_LeNet5)
print('The pruned LeNet-5 has %d nonzero parameters.\n'%nnz)

print('*** Testing the test accuracy of the pruned model...')
test_acc = test(model_LeNet5, test_x, test_y, device)
print('The test accuracy of the pruned LeNet-5 model is %.2f%%.\n'%(100*test_acc))

print('*** Testing layerwise sparsity ratio of the pruned model...')
calc_layerwise_sparsity(model_LeNet5)

print('\n*** Testing DNN_PALM_L0_Group_Lasso pruning algorithm on LeNet-5 ***')
print('----------------------------------------------------------------------------')

print('Loading a pruned LeNet-5 model...')
pruned_LeNet5_SparseGL_file = 'pruned_LeNet5_SparseGL_0.9915_nnz_2032.pkl'
model_LeNet5.load_state_dict(torch.load(pruned_LeNet5_SparseGL_file,map_location=device))#map_location=torch.device('cpu')
print('Pruned LeNet-5 model has been loaded successfully.\n')

print('*** Testing the number of nonzeros of the pruned model...')
nnz = cal_num_nonzero(model_LeNet5)
print('The pruned LeNet-5 has %d nonzero parameters.\n'%nnz)
print('*** Testing the architecture of the pruned model...')
print('The original structure is',[i for i in net_dims[:-1]])
neurons = remainingNeurons(model_LeNet5)
print('The pruned structure is',[i for i in neurons])

print('\n*** Testing the FLOPs of the pruned model...')
FLOPs = FLOPs_LeNet5_mnist(model_LeNet5)
print('The FLOPs of the pruned LeNet-5 model is %.2f%%.\n'%(100*FLOPs))

print('\n*** Testing the test accuracy of the pruned model...')
test_acc = test(model_LeNet5, test_x, test_y, device)
print('The test accuracy of the pruned LeNet-5 model is %.2f%%.\n'%(100*test_acc))

print('*** Testing layerwise sparsity ratio of the pruned LeNet-5 model obtained by DNN_PALM_L0_Group_Lasso pruning algo...')
calc_layerwise_sparsity(model_LeNet5)