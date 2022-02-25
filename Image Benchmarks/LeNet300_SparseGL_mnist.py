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

lam = 0.0004
l21 = 0.0002
increment = 1e-5
stride = 20
numRuns = 200
pruningEpochs = 10 # for proximal part
dnnEpochs = 30 # for retraining part
preepochs = 100
wd = 1e-5
gpu_id = 0
manual_seed = 1
cuda_manual_seed = 1
pruned_dir = 'pruned_models_results'
logname = 'log_LeNet300_SparseGL.txt'

# If you are using Colab, please comment the following lines and upload LeNet300_pretrained_0.9824.pkl
# In the leftmost part of Colab's interface, click 'Upload' and choose LeNet300_pretrained_0.9824.pkl on your computer.
import argparse
parser = argparse.ArgumentParser(description="Pruning LeNet-300-100")
parser.add_argument('--l0', type=float, default=4e-4, help="weight of L0 norm regularization(default=0.0001)")
parser.add_argument('--inc', type=float, default=1e-5, help="increment of L0 norm regularization(default=0.00001)")
parser.add_argument('--stride', type=int, default=20, help="weight decay step of L0 norm regularization(default=20)")
parser.add_argument('--l21', type=float, default=2e-4, help="weight of group lasso regularization")
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
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class Network(nn.Module):
    def __init__(self, net_dims, p=0, activation=nn.ReLU,dropout=nn.Dropout):
        """Constructor for multi-layer perceptron pytorch class
        params:
            * net_dims: list of ints  - dimensions of each layer in neural network
            * activation: func        - activation function to use in each layer
                                      - default is ReLU
        """
        super(Network, self).__init__()
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
              # m.bias.data.zero_()
              m.bias.data.uniform_(-1,1)

    def forward(self, x):
        loss = self.net(x)

        return loss

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

def Loss_GroupLasso(model):
    # to compute the loss of a model with sparse group lasso regularization
    Loss = 0
    for name,param in model.named_parameters():
        if 'weight' in name:
            weights = param.detach().clone()
            Loss += sum(weights.norm(dim=0))          
        else:
            bias = param.detach().clone()
            Loss += bias.norm(p=1)

    return Loss.item()

def GroupSparse_sol_forloop(u, lamb, eta,device):# accessible version
    # to get the solution to sparse group lasso regularization(proximal operator, Lemma 3.3 and Algo 4 in our paper)
    # We use for loops to make this function straightforward, but we end up with a slow implementation.
    S = torch.zeros_like(u)
    if len(u.shape)<2: # deal with biases
        S = torch.where(torch.abs(u)>lamb, 0.5*(u-lamb)**2-eta, S)# i is 1 since there is only one row when we deal with biases.
        u_new = torch.where(S>0, (1-lamb/torch.abs(u))*u, torch.zeros_like(u))        
    else:
        u_ones = torch.ones_like(u)
        u_zeros = torch.zeros_like(u)
        c_diag=torch.diag(torch.arange(0,u.shape[1])).to(device)# diagnal matrix 0-783
        cols = torch.mm(u_ones,c_diag.float())# indices for columns
        sorted_u, indices = torch.sort(torch.abs(u), 0, descending=True)      
        for i in range(S.shape[0]):# sweep the first i+1 rows
            y_i = sorted_u[:i+1,:].norm(dim=0)# column-wise L2 norm
            S[i,:] = torch.where(y_i>lamb, 0.5*(y_i-lamb)**2-(i+1)*eta, S[i,:])
        values, ind = torch.max(S, 0)#indices of maximum of each column of S        
        u_new = torch.zeros_like(u)
        for i in range(u.shape[1]):
            if values[i]>0:
                u_new[indices[:ind[i]+1,i],i] = u[indices[:ind[i]+1,i],i]        
        len_col_vec = u_new.norm(dim=0)# column-wise L2-norm 
        normalization_coef = torch.where(len_col_vec==0,torch.zeros_like(len_col_vec),1-lamb/len_col_vec)# deal with the case when denominator is zero, just get the coef, i.e. (1-lam/|y_k|)
        u_new = u_new*normalization_coef # scaling

    return u_new

def GroupSparse_sol(u, lamb, eta,device):# much faster than the for-loop version, but not accessible, b/c we use linear algebra tricks
    # to get the solution to sparse group lasso regularization(proximal operator, Lemma 3.3 and Algo 4 in our paper)
    # we use matrix operations instead of for loops to speed up our algo, which makes this function less accessible.
    S = torch.zeros_like(u)
    if len(u.shape)<2: # deal with biases
        S = torch.where(torch.abs(u)>lamb, 0.5*(u-lamb)**2-eta, S)# i is 1 since there is only one row when we deal with biases.
        u_new = torch.where(S>0, (1-lamb/torch.abs(u))*u, torch.zeros_like(u))        
    else:
        u_ones = torch.ones_like(u)
        u_zeros = torch.zeros_like(u)
        c_diag=torch.diag(torch.arange(0,u.shape[1])).to(device)# diagnal matrix 0-783
        cols = torch.mm(u_ones,c_diag.float())# indices for columns
        sorted_u, indices = torch.sort(torch.abs(u), 0, descending=True) 
        lower_tri = torch.tril(torch.ones(u.shape[0],u.shape[0])).to(device)#lower triangular matrix
        y_i = torch.sqrt(torch.mm(lower_tri,sorted_u**2))# y_i with l2 norm
        S = torch.where(y_i>lamb, 0.5*(y_i - lamb)**2 - eta*torch.mm(torch.diag(torch.arange(1,u.shape[0]+1)).to(device).float(),u_ones), S)# the else statement of the for loop in Algo 4 of our paper
        values, ind = torch.max(S, 0)#indices of maximum of each column of S
        indx = torch.where(values>0,ind+1,torch.zeros_like(ind))# a vector that stores ind+1 of the maximum of each column
        #indices of elements which should be kept in sorted. We set the entries in ind0 preceding the location of maximum of each col to be 1, otherwise 0.
        # If entries in ind0 are 1, that means the corresponding entries in sorted_u should be kept. Otherwise, they will be 0.
        ind0 = torch.where(torch.mm(u_ones,torch.diag(indx).to(device).float())<torch.mm(torch.diag(torch.arange(1,u.shape[0]+1)).to(device).float(),u_ones),u_zeros,u_ones)
        u_zeros[indices,cols.long()] = sorted_u*ind0 # restore the order of for each y_i and normalize it. We use u_zeros since we only care about the nonzero entries, which implies that we only need to fill out the locations whose entries are supposed to be 1.
        u_new_unnormalized = u_zeros*torch.sign(u)# restore signs since we sorted by magnitude previously
        len_col_vec = u_new_unnormalized.norm(dim=0)# column-wise L2-norm 
        normalization_coef = torch.where(len_col_vec==0,torch.zeros_like(len_col_vec),1-lamb/len_col_vec)# deal with the case when denominator is zero, just get the coef, i.e. (1-lam/|y_k|)
        u_new = u_new_unnormalized*normalization_coef # scaling

    return u_new

def test(model,val_loader,device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)

        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    test_acc = 100.*correct/total
    # print('Test acc: %.2f' % test_acc)
    return test_acc

def retrain(model, train_set, val_loader, wd, dnnepochs, step, device):
    print('--- Retraining step ', step)
    mask_list = []
    for p in model.parameters():
      mask_list.append( torch.where(p.detach()!=0, torch.ones_like(p.detach()), torch.zeros_like(p.detach())) )

    model.net[0].weight.register_hook(lambda grad: grad.mul_(mask_list[0]))
    model.net[0].bias.register_hook(lambda grad: grad.mul_(mask_list[1]))
    model.net[3].weight.register_hook(lambda grad: grad.mul_(mask_list[2]))
    model.net[3].bias.register_hook(lambda grad: grad.mul_(mask_list[3]))
    model.net[6].weight.register_hook(lambda grad: grad.mul_(mask_list[4]))
    model.net[6].bias.register_hook(lambda grad: grad.mul_(mask_list[5]))

    optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=wd)
    best_acc = [0]
    for epoch in range(dnnepochs):
        for x, y in train_set:
            model.train()
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        nnz = cal_num_nonzero(model)
        # test_output = model(testData)
        # pred_y = torch.max(test_output, 1)[1].data#.numpy()
        # accuracy = float((pred_y == test_y.data).sum()) / float(test_y.size(0))#.astype(int)        
        accuracy = test(model,val_loader,device)
        if accuracy > best_acc[-1]:
            best_acc.append(accuracy)
            fileName = os.path.join(pruned_dir, 'pruned_LeNet300GL_retr_' + str(accuracy) + 'nnz_' + str(num_nz[-1]) + '.pkl')
            torch.save(model.state_dict(), fileName) 
        print('Epoch: ', epoch+1, '| test accuracy: %.4f' % accuracy,'|nnz:',nnz,'lam:', lam,'|wd:',wd,'|l21',l21)
    model.net[0].weight.register_hook(lambda grad: grad)
    model.net[0].bias.register_hook(lambda grad: grad)
    model.net[3].weight.register_hook(lambda grad: grad)
    model.net[3].bias.register_hook(lambda grad: grad)
    model.net[6].weight.register_hook(lambda grad: grad)
    model.net[6].bias.register_hook(lambda grad: grad)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net_dims = [784,300,100,10]
model = Network(net_dims)#,p=args.dropout
# testData = test_x.view(test_x.shape[0], -1)
train_set = [ (x.view(x.shape[0], -1),y) for x,y in train_loader]
val_loader = [ (x.view(x.shape[0], -1),y) for x,y in val_loader]
if device.type != 'cpu':
    torch.cuda.set_device(gpu_id)
    torch.cuda.manual_seed(cuda_manual_seed)
    print('put tensors onto cuda')
    model = model.to(device)
    train_set = [ (x.view(x.shape[0], -1),y) for x,y in train_loader]
    # testData, test_y = testData.to(device), test_y.to(device)

optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=wd)
import os.path
param_file = 'LeNet300_pretrained_0.9824.pkl' # pretrained model has a test accuracy of 98.24%
if os.path.isfile(param_file):
    print("Pre-trained model exists:" + param_file)    
    model.load_state_dict(torch.load(param_file,map_location=torch.device(device)))
    text = param_file[param_file.startswith("LeNet300_pretrained_") and len("LeNet300_pretrained_"):]
    text = text[:text.endswith(".pkl") and -len(".pkl")]
    base_test_acc = float(text)    
else:
    print("Pre-trained model does not exist, so before pruning we have to pre-train a model.")
    for epoch in range(preepochs):
        for x, y in train_set:
            model.train()
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = test(model,val_loader,device)
        print('Epoch: ' + str(epoch+1) + '| test accuracy: %.4f' % acc)        
    base_test_acc = acc
    torch.save(model.state_dict(), 'LeNet300_pretrained_'+str(base_test_acc)+'.pkl')

print('base acc:',base_test_acc)

for step in range(numRuns):
    if (step+1)%stride==0:
        lam += increment
        l21 += increment
    print('Sparse GroupLasso step', step, 'l0:',lam,'l21:',l21,'wd:',wd,'numRuns:',numRuns, 'inc:',increment,'stride:',stride)
    L_list = []
    num_nz = []
    i = 0
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_struc = [0] # store the pruned structure with the best test accuracy in this run
    old_model_para = [p.detach().clone() for p in model.parameters()]
    for epoch in range(pruningEpochs):
        model.train()
        for x, y in train_set:
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
                    weights.data = GroupSparse_sol((old_w-s*grad_w).to(device),l21*s,lam*s,device)# to apply the proximal operator to solve it(Algo 4 in our paper)
                    biases.data = GroupSparse_sol((old_b-s*grad_b).to(device),l21*s,lam*s,device)                
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
                num_nz.append(cal_num_nonzero(model))
                lossl_batch = loss.data+lam*num_nz[-1]+l21*Loss_GroupLasso(model)
                accuracy = test(model,val_loader,device)
                neurons = remainingNeurons(model)
                flops = cal_FLOPs(model)
                if accuracy >= best_struc[-1]:
                    best_struc.append(calc_layerwise_nonzeros(model))
                    best_struc.append(flops)            
                    best_struc.append(neurons)
                    best_struc.append(num_nz[-1])
                    best_struc.append(accuracy)
                    fileName = os.path.join(pruned_dir, 'pruned_LeNet300GL_' + str(accuracy) + 'nnz_' + str(num_nz[-1]) + '.pkl')
                    torch.save(model.state_dict(), fileName)                    
                print('Epoch: ', epoch+1, '| train loss: %.3f' % lossl_batch,\
                 '| test accuracy: %.4f' % accuracy, '|nnz: %d' %num_nz[-1],\
                  '|neurons: ',[i for i in neurons],'L:',L,'|FLOPs:%.4f'%flops)
    
    print('Base test acc:%.4f.'%base_test_acc,'After L0 Group Lasso pruning, pruned stucture with best test acc, acc: %.4f' % best_struc[-1],\
     'nnz:%d' % best_struc[-2], 'neurons:' , [i for i in best_struc[-3]],'|FLOPs:%.4f'%best_struc[-4])
    print('sparsity_ratio',best_struc[-5])
    model = retrain(model, train_set, val_loader, wd, dnnEpochs, step, device)