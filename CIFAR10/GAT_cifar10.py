#torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data.sampler import SubsetRandomSampler

# torch dependencies for data load 
import torchvision
from torchvision import datasets, transforms
# numpy
import numpy as np
# time
import time
import random
#import scipy.stats as stats

######################parse inputs###################
import getopt
import sys


import os
if not os.path.isdir('./results'):
    os.mkdir('./results')
if not os.path.isdir('./log'):
    os.mkdir('./log')
if not os.path.isdir('./models'):
    os.mkdir('./models')
if not os.path.isdir('./data'):
    os.mkdir('./data')


#READ ARGUMENTS
opts = sys.argv[1::2]
args = sys.argv[2::2]

#Defaults
l_ce = 1.0
l2_reg = 10.0
mul = 4.0
TRAIN_BATCH_SIZE = 64
Feps = 8.0
B_val = 4.0
MAX_EPOCHS = 100
lr_factor = 10.0


for  i in range(len(opts)):
    opt = opts[i]
    arg = args[i]
    #Experiment name
    if opt=='-EXP_NAME':
        EXP_NAME = str(arg)
        LOG_FILE_NAME = 'log/'+str(arg)+'.txt'
        print('EXP_NAME:',EXP_NAME)
    if opt=='-MAX_EPOCHS':
        MAX_EPOCHS = int(arg)
        print('MAX_EPOCHS:',MAX_EPOCHS)
    if opt=='-l_ce':
        l_ce = float(arg)
        print('l_ce:',l_ce)
    if opt=='-B_val':
        B_val = float(arg)
        print('Initial Noise Magnitude:',B_val)
    if opt=='-l2_reg':
        l2_reg = float(arg)
        print('l2_reg:',l2_reg)
    if opt=='-b_size':
        TRAIN_BATCH_SIZE = int(arg)
        print('Training Batch Size:',TRAIN_BATCH_SIZE)
    if opt=='-Feps':
        Feps = float(arg)
        print('RFGSM Epsilon:',Feps)
    if opt=='-lr_factor':
        lr_factor = float(arg)
        print('lr_factor:',lr_factor)
    if opt=='-mul':
        mul = float(arg)
        print('Step Mult. factor:',mul)
    

###################################### Function Definitions #######################################
def FGSM_Attack_step(model,loss,image,target,eps=0.1,bounds=[0,1],GPU=0,steps=30): 
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps 
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img
   

def execfile(filepath):
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'))
        globals().update(locals())
   
#######################################Cudnn##############################################
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True 
print('Cudnn status:',torch.backends.cudnn.enabled)
#######################################Set tensor to CUDA#########################################
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#######################################Parameters##################################################
TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
VAL_BATCH_SIZE   = 128
TEST_BATCH_SIZE   = 128
BASE_LR          = 1e-1
MAX_ITER         = (MAX_EPOCHS*50000)/TRAIN_BATCH_SIZE
MODEL_PREFIX     = 'models/' + EXP_NAME + '_'
#######################################load network################################################
execfile('ResNet.py')
model = ResNet18()
model.cuda()
model.train()
######################################Load data ###################################################
transform_train = transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

transform_test = transforms.Compose([
        transforms.ToTensor(),])

train_set  = torchvision.datasets.CIFAR10(root='./data', train=True , download=True, transform=transform_train)
val_set    = torchvision.datasets.CIFAR10(root='./data', train=True , download=True, transform=transform_test)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Creating Train-Val split from original Training set
train_size = 49000
valid_size = 1000
test_size  = 10000

train_indices = list(range(50000))
val_indices = []
count = np.zeros(10)
for index in range(len(train_set)):
    _, target = train_set[index]
    if(np.all(count==100)):
        break
    if(count[target]<100):
        count[target] += 1
        val_indices.append(index)
        train_indices.remove(index)
        
print("Size of train set:",len(train_indices))
print("Size of val set:",len(val_indices))
#get data loader for train, val and test
train_loader = torch.utils.data.DataLoader(train_set,batch_size=TRAIN_BATCH_SIZE ,sampler=SubsetRandomSampler(train_indices))
val_loader   = torch.utils.data.DataLoader(val_set,sampler = SubsetRandomSampler(val_indices),batch_size=VAL_BATCH_SIZE)
test_loader   = torch.utils.data.DataLoader(test_set,batch_size=TEST_BATCH_SIZE)
print('CIFAR10 dataloader: Done')  
###################################################################################################
epochs    = MAX_EPOCHS
iteration = 0
loss      = nn.CrossEntropyLoss()
LR = BASE_LR
optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9,weight_decay=5e-4)
optimizer.zero_grad()


##################################################################################################
for epoch in range(epochs):
    start = time.time()
    iter_loss =0 
    counter =0 
    
    for data, target in train_loader:
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
    
        qdata = data + ((B_val/255.0)*torch.sign(torch.tensor([0.5]) - torch.rand_like(data)).cuda())
        qdata = torch.clamp(qdata,0.0,1.0)
        
        model.eval()
        qadv = FGSM_Attack_step(model,loss,qdata,target,eps=Feps/255.0,steps=1)

        delta = qadv - data
        delta = torch.clamp(delta,-8.0/255.0,8.0/255)
        qadv = data+delta
        qadv = torch.clamp(qadv,0.0,1.0)
        
        model.train()
        optimizer.zero_grad()
        qout  = model(qadv)
        out  = model(data)
        
        Q_out = nn.Softmax(dim=1)(qout)
        P_out = nn.Softmax(dim=1)(out)
        
        '''LOSS COMPUTATION'''
        
        
        closs = loss(out,target)
        
        reg_loss =  ((P_out - Q_out)**2.0).sum(1).mean(0)
        
        cost = l_ce*closs + l2_reg*reg_loss
        cost.backward()
        optimizer.step()
        
        if iteration%100==0:
            msg = 'iter,'+str(iteration)+',clean loss,'+str(closs.data.cpu().numpy()) \
            +',reg loss,'+str(reg_loss.data.cpu().numpy()) \
            +',total loss,'+str(cost.data.cpu().numpy()) \
                                        +'\n'
            log_file = open(LOG_FILE_NAME,'a+')
            log_file.write(msg)
            log_file.close()
            model.train()
            #print msg
        iteration = iteration + 1
        ##console log
        counter = counter + 1
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] : Loss:%f \t\t'
                %(epoch, MAX_EPOCHS, counter,
                    (train_size/TRAIN_BATCH_SIZE),cost.data.cpu().numpy()))
    end = time.time()
    print('Epoch:',epoch,' Time taken:',(end-start))
    
    model_name = MODEL_PREFIX+str(epoch)+'.pkl'
    torch.save(model.state_dict(),model_name)
 
    if epoch in [70,85]:
        LR /= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
    if epoch == 85:
        l2_reg = l2_reg*mul




#######################################################################################################################
model.eval()


def PGD(model,loss,data,target,eps=0.1,eps_iter=0.1,bounds=[],steps=1):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = torch.clamp(noise,-eps,eps)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        #compute loss using true label
        cost = loss(out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
    img = data + noise
    return img


##########################FIND BEST MODEL###############################################
EVAL_LOG_NAME = 'results/'+EXP_NAME+'.txt'
ACC_EPOCH_LOG_NAME = 'results/'+EXP_NAME+'acc_epoch.txt'
ACC_IFGSM_EPOCH_LOG_NAME = 'results/'+EXP_NAME+'ifgsm_acc_epoch.txt'
log_file = open(EVAL_LOG_NAME,'a+')
msg = '##################### iter.FGSM: steps=7,eps=8.0/255,1####################\n'
log_file.write(msg)
log_file.close()
accuracy_log = np.zeros(MAX_EPOCHS)
for epoch in range(MAX_EPOCHS):
    model_name = MODEL_PREFIX+str(epoch)+'.pkl'
    model.load_state_dict(torch.load(model_name))
    eps=8.0/255
    accuracy = 0
    accuracy_ifgsm = 0
    i = 0
    for data, target in val_loader:
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum()
        i = i + 1 
    for data, target in val_loader:
        data = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=7)
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy_ifgsm = accuracy_ifgsm + prediction.eq(target.data).sum()
    acc = (accuracy.item()*1.0) / (i*VAL_BATCH_SIZE) * 100
    acc_ifgsm = (accuracy_ifgsm.item()*1.0) / (i*VAL_BATCH_SIZE) * 100
    #log accuracy to file
    msg= str(epoch)+','+str(acc)+'\n'
    log_file = open(ACC_EPOCH_LOG_NAME,'a+')
    log_file.write(msg)
    log_file.close()
    
    msg1= str(epoch)+','+str(acc_ifgsm)+'\n'
    log_file = open(ACC_IFGSM_EPOCH_LOG_NAME,'a+')
    log_file.write(msg1)
    log_file.close()

    accuracy_log[epoch] = acc_ifgsm
    sys.stdout.write('\r')
    sys.stdout.write('| Epoch [%3d/%3d] : Acc:%f \t\t'
            %(epoch, MAX_EPOCHS,acc))
    sys.stdout.flush() 

log_file = open(EVAL_LOG_NAME,'a+')
msg = 'Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
log_file.write(msg)
log_file.close()

model_name = MODEL_PREFIX+str(accuracy_log.argmax())+'.pkl'
model.load_state_dict(torch.load(model_name))
model.eval()
model.cuda()
##################################### FGSM #############################################
EVAL_LOG_NAME = 'results/'+EXP_NAME+'.txt'
log_file = open(EVAL_LOG_NAME,'a+')
msg = '##################### FGSM ####################\n'
log_file.write(msg)
log_file.close()
for eps in np.arange(0.0/255,10.0/255,2.0/255):
    i = 0
    accuracy = 0
    for data, target in test_loader:
        adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=1)
        data   = Variable(adv).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum()
        i = i + 1
    acc = (accuracy.item()*1.0) / (test_size) * 100
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
    log_file.write(msg)
    log_file.close()
##################################### iFGSM #############################################
log_file = open(EVAL_LOG_NAME,'a+')
msg = '##################### iFGSM: step=7 ####################\n'
log_file.write(msg)
log_file.close()
for eps in np.arange(2.0/255,10.0/255,2.0/255):
    i = 0
    accuracy = 0
    for data, target in test_loader:
        adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=7)
        data   = Variable(adv).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum()
        i = i + 1
    acc = (accuracy.item()*1.0) / (test_size) * 100
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
    log_file.write(msg)
    log_file.close()



def MSPGD(model,loss,data,target,eps=0.1,eps_iter=0.1,bounds=[],steps=[7,20,50,100,500]):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = torch.clamp(noise,-eps,eps)
    img_arr = []
    for step in range(steps[-1]):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        #compute loss using true label
        cost = loss(out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
        for j in range(len(steps)):
            if step == steps[j]-1:
                img_tmp = data + noise
                img_arr.append(img_tmp)
                break
    return img_arr

##################################### PGD, steps=[7,20,50,100,500] #############################################
log_file = open(EVAL_LOG_NAME,'a+')
msg = '##################### PGD: steps=[7,20,50,100,500],eps_iter=2/255 ####################\n'
log_file.write(msg)
log_file.close()
all_steps = [7,20,50,100,500] 
num_steps = len(all_steps)
eps = 8.0/255
i = 0
acc_arr = torch.zeros((num_steps))
for data, target in test_loader:
    adv_arr = MSPGD(model,loss,data,target,eps=eps,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps)     
    target = Variable(target).cuda()
    for j in range(num_steps):
        data   = Variable(adv_arr[j]).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum()
    i = i + 1
print(acc_arr)
for j in range(num_steps):
    acc_arr[j] = (acc_arr[j].item()*1.0) / (test_size) * 100
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'eps,'+str(eps)+',steps,'+str(all_steps[j])+',Acc,'+str(acc_arr[j])+'\n'
    log_file.write(msg)
    log_file.close()
