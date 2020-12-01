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

######################parse inputs###################
import getopt
import sys,os
#READ ARGUMENTS
opts = sys.argv[1::2]
args = sys.argv[2::2]
#print opts
#print args


for  i in range(len(opts)):
    opt = opts[i]
    arg = args[i]
    #Experiment name
    if opt=='-EXP_NAME':
        EXP_NAME = str(arg)
        print('EXP_NAME:',EXP_NAME)
    if opt=='-MODEL':
        MODEL = str(arg)
        print('MODEL:',MODEL)


#######################################Cudnn##############################################
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True 
print('Cudnn status:',torch.backends.cudnn.enabled)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#######################################load network################################################
sys.path.append("./")
from arch import arch
model = arch() 
model.load_state_dict(torch.load(MODEL))
model = torch.nn.DataParallel(model)
model.cuda()
model.eval()

######################################Load data ###################################################
transform_test = transforms.Compose([
        transforms.ToTensor(),])
test_set   = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
                          
test_size  = 10000
batch_size = 300
test_loader   = torch.utils.data.DataLoader(test_set,batch_size=batch_size)
print('CIFAR10 dataloader: Done')

#######################################################################################################################
CE_loss      = nn.CrossEntropyLoss()

#######################################################################################################################
EVAL_LOG_NAME = EXP_NAME+'.txt'

#######################################################################################################################
def max_margin_loss(x,y):
    B = y.size(0)
    corr = x[range(B),y]

    x_new = x - 1000*torch.eye(10)[y].cuda()
    tar = x[range(B),x_new.argmax(dim=1)]
    loss = tar - corr
    loss = torch.mean(loss)
    
    return loss


def GAMA_PGD(model,data,target,eps,eps_iter,bounds,steps,w_reg,lin,SCHED,drop):

    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        orig_out = model(orig_img)
        P_out = nn.Softmax(dim=1)(orig_out)
        
        out  = model(img)
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if step <= lin:
            cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out,tar)
            W_REG -= w_reg/lin
        else:
            cost = max_margin_loss(Q_out,tar)
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

    return data + noise
    
STEPS = 100
RR = 1
SCHED = [60,85]
drop = 10    
lin = 25
w_reg = 50
msg = '\n################ GAMA-PGD Attack with '+str(RR)+' Random Restarts ################\n'
print(msg)
log_file = open(EVAL_LOG_NAME,'a+')
log_file.write(msg)
log_file.close()
eps = 8.0/255.0

accuracy = np.zeros(RR)
worst_accuracy = np.zeros(RR)
tcost = np.zeros(RR)
test_data = 0
for data, target in test_loader:
    worst = torch.ones(target.size()[0]).type(torch.ByteTensor).cuda()
    target = Variable(target).cuda()
    track = np.ones(RR)
    for r in range(RR):
        adv = GAMA_PGD(model,data,target,eps=eps,eps_iter=16.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=STEPS,w_reg=w_reg,lin=lin,SCHED=SCHED,drop=drop)   
        adv = Variable(adv).cuda()
        out = model(adv)
        cost = CE_loss(out,target)
        prediction = out.data.max(1)[1]
        if(r>=1):
            track[r-1] = 0
        worst = torch.mul(worst,prediction.eq(target.data))
        tcost += track*(cost.data.cpu().numpy()*target.size()[0])
        accuracy += track*prediction.eq(target.data).sum().item()
        worst_accuracy[r] += worst.sum().item()
    test_data += target.size()[0]
    print(test_data)
acc = ((accuracy*100.0) / float(test_data))/(np.arange(1,RR+1,1))
tcost = (tcost / float(test_data))/(np.arange(1,RR+1,1))
worst_accuracy = 100*worst_accuracy/float(test_data)
for i in range(1,RR+1,1):
    msg = 'Restart '+str(i)+', steps '+str(STEPS)+' ,Average Acc:'+str(acc[i-1])+', Worst Acc:'+str(worst_accuracy[i-1])+',CE loss,'+str(tcost[i-1])+'\n'
    log_file = open(EVAL_LOG_NAME,'a+')
    log_file.write(msg)
    log_file.close()