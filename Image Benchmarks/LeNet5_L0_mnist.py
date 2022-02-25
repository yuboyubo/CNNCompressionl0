# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.utils.data as Data
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import argparse
import os

lam = 0.01
increment = 0.001
stride = 30
numRuns = 200
pruningEpochs = 10 # for proximal part
dnnEpochs = 30 # for retraining part
preepochs = 100
wd = 1e-5
gpu_id = 0
manual_seed = 1
cuda_manual_seed = 1
pruned_dir = 'pruned_models_results'
logname = 'log_LeNet5_L0.txt'

# If you are using Colab, please comment the following lines and upload LeNet300_pretrained_0.9824.pkl
# In the leftmost part of Colab's interface, click 'Upload' and choose LeNet300_pretrained_0.9824.pkl on your computer.
import argparse
parser = argparse.ArgumentParser(description="Pruning LeNet-5")
parser.add_argument('--l0', type=float, default=1e-1, help="weight of L0 norm regularization(default=0.1)")
parser.add_argument('--inc', type=float, default=1e-2, help="increment of L0 norm regularization(default=0.01)")
parser.add_argument('--stride', type=int, default=50, help="weight decay step of L0 norm regularization(default=50)")
parser.add_argument('--l21', type=float, default=1e-4, help="weight of group lasso regularization")
parser.add_argument('--wd', type=float, default=1e-5, help="weight decay")
parser.add_argument('--pruningepochs', type=int, default=10, help="num of epochs")
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--ini', action='store_true', default=True, help='kaiming init')
parser.add_argument('--gpu', type=int, default=0, help="which gpu are you going to use?(default=0)")
parser.add_argument('--numruns', type=int, default=200, help="alternation times (default: 200)")
parser.add_argument('--dnnepochs', type=int, default=30, help="DNN epochs (default: 10)")
parser.add_argument('--dropout', type=float, default=0, help="the probability for dropout(default=0)")
parser.add_argument('--preepochs', type=int, default=100, help="Pretraining epochs (default: 100)")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pruned-dir', dest='pruned_dir',
                    help='The directory used to save the pruned models',
                    default='pruned_models_results', type=str)
parser.add_argument('--logname', default='log_LeNet300_L0.txt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()
lam = args.l0
stride = args.stride
increment = args.inc
numRuns = args.numruns
pruningEpochs = args.pruningepochs # for proximal part
dnnEpochs = args.dnnepochs # for retraining part
preepochs = args.preepochs
wd = args.wd
manual_seed = args.seed
gpu_id = args.gpu
cuda_manual_seed = args.seed
pruned_dir = args.pruned_dir
logname = args.logname
# Comment the above lines if you are using Colab.

torch.manual_seed(manual_seed)

if not os.path.exists(pruned_dir):
    os.makedirs(pruned_dir)
logname=os.path.join(pruned_dir, logname)
if not os.path.isfile(logname):
    with open(logname, 'w'): pass

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

# To save time we only test the first 2000 samples
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels

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

def hard_thresholding(u,theta):
    u = torch.where(torch.abs(u) > theta, u, torch.zeros_like(u))
    return u

def cal_num_nonzero(model):
    # calculate the number of nonzero parameters in a model
    cnt = 0
    for param in model.parameters():
        cnt += param.detach().nonzero().size(0)

    return cnt

def calc_layerwise_nonzeros(model):
    sparsity_ratio = []
    for name,param in model.named_parameters():
        nonzeros = param.detach().norm(0).item()    
        num_params = param.numel()
        sparsity_ratio.append([name, nonzeros,'/', num_params,'(%.2f%%)' % (nonzeros/num_params*100)])

    return sparsity_ratio

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

def FLOPs_LeNet5_mnist(model):
    '''
    Input: 1x28*28
    Conv1: 20*1*5*5(kernel) -> 20*24*24(Conv1 and Relu) -> 20*12*12(MaxPooling)
    Conv2: 50*20*5*5(kernel)-> 50* 8* 8(Conv2 and Relu) -> 50* 4* 4(MaxPooling)
    Flatten: 50* 4* 4 = 800
    Fc1:     500*800 (no Relu here since we have used Relu after Conv2)
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

def retrain(model, train_loader, test_x,test_y, wd, dnnepochs, step, device):      
    print('--- Retraining step ', step)
    mask_list = []
    for p in model.parameters():
      mask_list.append( torch.where(p.detach()!=0, torch.ones_like(p.detach()), torch.zeros_like(p.detach())) )

    model.conv1.weight.register_hook(lambda grad: grad.mul_(mask_list[0]))
    model.conv1.bias.register_hook(lambda grad: grad.mul_(mask_list[1]))
    model.conv2.weight.register_hook(lambda grad: grad.mul_(mask_list[2]))
    model.conv2.bias.register_hook(lambda grad: grad.mul_(mask_list[3]))
    model.fc1.weight.register_hook(lambda grad: grad.mul_(mask_list[4]))
    model.fc1.bias.register_hook(lambda grad: grad.mul_(mask_list[5]))
    model.fc2.weight.register_hook(lambda grad: grad.mul_(mask_list[6]))
    model.fc2.bias.register_hook(lambda grad: grad.mul_(mask_list[7]))

    optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=wd)
    best_acc = [0]
    nnz = cal_num_nonzero(model)
    for epoch in range(dnnepochs):
        for x, y in train_loader:
            model.train()
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()            
        losslist.append(loss.data)
        test_output = model(test_x)
        pred_y = torch.max(test_output, 1)[1].data#.numpy()
        acc = float((pred_y == test_y.data).sum()) / float(test_y.size(0))#.astype(int)
        if acc > best_acc[-1]:
            best_acc.append(acc)
            if acc>=0.991 and nnz<2500: # only save expected models
                fileName = os.path.join(pruned_dir, 'pruned_LeNet5_L0_retr_' + str(acc) + 'nnz_' + str(nnz) + '.pkl')
                torch.save(model.state_dict(), fileName)
        print('Epoch: ', epoch+1, '| test accuracy: %.4f' % acc,\
        '|nnz:',nnz,'lam:', lam,'|wd:',wd)
    print('Best acc after retraining:', best_acc[-1])
    model.conv1.weight.register_hook(lambda grad: grad)
    model.conv1.bias.register_hook(lambda grad: grad)
    model.conv2.weight.register_hook(lambda grad: grad)
    model.conv2.bias.register_hook(lambda grad: grad)
    model.fc1.weight.register_hook(lambda grad: grad)
    model.fc1.bias.register_hook(lambda grad: grad)
    model.fc2.weight.register_hook(lambda grad: grad)
    model.fc2.bias.register_hook(lambda grad: grad)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = LeNet5().to(device)
test_x, test_y = test_x.to(device), test_y.to(device)
acclist = []
optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=wd)
losslist = []
# for epoch in range(args.preepochs):
#     for x, y in train_loader:
#         model.train()
#         x, y = x.to(device), y.to(device)
#         out = model(x)
#         loss = nn.CrossEntropyLoss()(out, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     model.eval()          
#     losslist.append(loss.data)
#     test_output = model(test_x)
#     pred_y = torch.max(test_output, 1)[1].data#.numpy()
#     base_test_acc = float((pred_y == test_y.data).sum()) / float(test_y.size(0))#.astype(int)
#     acclist.append(base_test_acc)
#     print('Epoch: ', epoch+1, '| train loss: %.4f' % losslist[-1], '| test accuracy: %.4f' % base_test_acc)

param_file = 'LeNet5_pretrained_0.9912.pkl' # load a pretrained model
model.load_state_dict(torch.load(param_file,map_location=torch.device(device)))
text = param_file[param_file.startswith("LeNet5_pretrained_") and len("LeNet5_pretrained_"):]
text = text[:text.endswith(".pkl") and -len(".pkl")]
base_test_acc = float(text)
print('base acc:',base_test_acc)
# torch.save(model.state_dict(), 'LeNet_params'+str(base_test_acc)+'.pkl')

for step in range(numRuns):
    if (step+1)%stride==0:
        lam += increment
    model.train()    
    L_list = []
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_struc = [0]    
    i = 0
    old_model_para = [p.detach().clone() for p in model.parameters()]
    print('--- Lenet5 L0 step ', step,'l0:',lam,'wd:',wd,'numRuns:',numRuns, 'increment:',increment,'stride:',stride)
    for epoch in range(pruningEpochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            i += 1
            old_model_para = [p.detach().clone() for p in model.parameters()]
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)            
            for (name,param), old_param in zip(model.named_parameters(), old_model_para):
                if 'weight' in name:
                    weights = param
                    old_w = old_param
                    continue
                else:
                    biases = param
                    old_b = old_param
                optimizer.zero_grad()
                loss.backward()
                fy = loss.detach().clone()
                grad_w = weights.grad
                grad_b = biases.grad
                L = 1
                for k in range(100):                
                    s = 1.0/(1.1*L)
                    weights.data = hard_thresholding((old_w-s*grad_w).to(device),lam*s)
                    biases.data = hard_thresholding((old_b-s*grad_b).to(device),lam*s)                
                    subtr_w = torch.sub(weights.detach(),old_w)#subtr_w = torch.sub(weights.data,old_w)
                    subtr_b = torch.sub(biases.detach(),old_b)#subtr_b = torch.sub(biases.data,old_b)
                    # Q_L = fy + torch.sum( torch.mul( subtr,grad_values ) ) + L/2*torch.sum( torch.pow(subtr,2) )
                    Q_L = fy + torch.sum( torch.mul( subtr_w,grad_w ) ) + torch.sum( torch.mul( subtr_b,grad_b ) ) + L/2*torch.sum( torch.pow(subtr_w,2) ) + L/2*torch.sum( torch.pow(subtr_b,2) )
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    if loss.data <= Q_L:
                        L_list.append(L)
                        break
                    L = 1.1*L
            
            if i % 600 == 0:
                model.eval() 
                nnz = cal_num_nonzero(model)
                loss_batch = loss.data+lam*nnz
                test_output = model(test_x)
                pred_y = torch.max(test_output, 1)[1].data#.numpy()
                acc = float((pred_y == test_y.data).sum()) / float(test_y.size(0))
                neurons = remainingNeurons(model)
                flops = FLOPs_LeNet5_mnist(model)
                if acc >= best_struc[-1]:
                    best_struc.append(calc_layerwise_nonzeros(model))
                    best_struc.append(flops)
                    best_struc.append(neurons)
                    best_struc.append(nnz)
                    best_struc.append(acc)
                    if acc>=0.991 and nnz<2500: # only save expected models
                        fileName = os.path.join(pruned_dir, 'pruned_LeNet5_L0_' + str(acc) + 'nnz_' + str(nnz) + '.pkl')
                        torch.save(model.state_dict(), fileName)
                print('Epoch: ', epoch+1, '| train loss: %.3f' % loss_batch, '| test accuracy: %.4f' % acc,\
                 '|nnz: %d' %nnz, '|neurons: ',[i for i in neurons], '|FLOPs:%.4f'%flops,'L:',L)    

    print('Base test acc:%.4f.'%base_test_acc,'After LeNet5 L0 pruning, pruned stucture with best test acc, acc: %.4f' % best_struc[-1],\
     'nnz:%d' % best_struc[-2], 'neurons:' , [i for i in best_struc[-3]],'|FLOPs:%.4f'%best_struc[-4])
    print('sparsity_ratio',best_struc[-5])
    model = retrain(model, train_loader, test_x,test_y, wd, dnnEpochs, step, device)
