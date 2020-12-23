""" 

This code is adapted from the following: https://github.com/pytorch/examples/tree/master/imagenet

"""

import argparse
import os
import random
import shutil
import time
import math
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

def execfile(filepath):
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'))
        globals().update(locals())
 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
if not os.path.isdir('./models'):
    os.mkdir('./models')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--lce', default=1.0, type=float,
                    help='Hyperparam for ce loss.')
                    
parser.add_argument('--START_EVAL', default=0, type=int,
                    help='Start PGD only from this epoch, skip early models ')
                    
parser.add_argument('--EXP_NAME', default="GAT", type=str,
                    help='Experiment Name ')
                    
parser.add_argument('--l2_reg', default=20, type=float,
                    help='Hyperparam for l2 regularizer.')
                    
parser.add_argument('--Bval', default=4.0, type=float,
                    help='Hyperparam for initial Bernoulli noise magnitude')
parser.add_argument('--Feps', default=8.0, type=float,
                    help='Hyperparam for FGSM Epsilon Magnitude.')
                    
parser.add_argument('--mul', default=7.0, type=float,
                    help='Hyperparam for step increase of l2 reg.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

###### PGD EVAL PARAMS  ######
steps = 20
eps = 8.0/float(255)
eps_iter = 2.5*eps/float(steps)


Clean_Acc = []
PGD_Acc = []

args = parser.parse_args()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    print("world_size:",args.world_size)
    print("mp:",args.multiprocessing_distributed)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("dist:",args.distributed)

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus:",ngpus_per_node)
    print("gpu:",args.gpu)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
        
        print("\n\n\n################### PGD-"+str(steps)+" EVAL ###################\n\n")
        print("Starting PGD Evaluation from Epoch",args.START_EVAL)
        for epoch in range(args.START_EVAL,args.epochs):
            args.model = 'models/'+args.EXP_NAME+'_checkpoint_'+str(epoch)+'.pth.tar'
            main_pgd_worker(args.gpu, ngpus_per_node, args)
            
    print("\n\n\nValidation Acc",Clean_Acc)
    print("\n\n\nPGD Acc",PGD_Acc)
    P = np.array(PGD_Acc)
    best_epoch = args.START_EVAL+P.argmax()
    msg = '\n\n Best Epoch:'+str(best_epoch)+' ,   PGD Acc:'+str(P.max())+'\n'
    print(msg)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True,num_classes=100)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False,num_classes=100)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']+1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    print(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        Clean_Acc.append(acc1.item())

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, filename = 'models/'+str(args.EXP_NAME)+'_checkpoint_'+str(epoch)+'.pth.tar')
            


def main_pgd_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=False,num_classes=100)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            if args.gpu is None:
                checkpoint = torch.load(args.model)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.model, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
            sys.exit()

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # evaluate on validation set
    acc1 = pgdvalidate(val_loader, model, criterion, args)
    PGD_Acc.append(acc1.item())

    
def FGSM_Attack_step(args,model,loss,image,target,eps=0.1,bounds=[0,1],GPU=0,steps=30): 
    tar = Variable(target.cuda(args.gpu, non_blocking=True))
    img = image.cuda(args.gpu, non_blocking=True)
    eps = eps/steps 
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda(args.gpu, non_blocking=True) 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img
    
    
    
def Guided_Attack(args,model,loss,image,target,eps=8/255,bounds=[0,1],steps=1,data=[],l2_reg=5,alt=1,B=64): 
    tar = Variable(target.cuda(args.gpu, non_blocking=True))
    img = image.cuda(args.gpu, non_blocking=True)
    eps = eps/steps 
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(data)
        rout  = model(img)

        P_out = nn.Softmax(dim=1)(out)
        R_out = nn.Softmax(dim=1)(rout)
        cost = loss(rout,tar) + alt*l2_reg*(((P_out - R_out)**2.0).sum(1)).mean(0) 
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda(args.gpu, non_blocking=True) 
    return adv
    
    
    
def pgd(args,model,loss,data,target,eps,eps_iter,bounds=[],steps=1):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    # Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target)
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(B,C,H,W).uniform_(-eps,eps).cuda(args.gpu, non_blocking=True)
    #pgd_target = torch.randint(0,1000,tar.size()).cuda(args.gpu, non_blocking=True)
    #while pgd_target.eq(target).any():
    #    idxs = pgd_target.eq(target).nonzero()
    #    pgd_target[idxs] = torch.randint(0,1000,idxs.size()).cuda(args.gpu, non_blocking=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        # compute loss using true label
        cost = loss(out,target)
        #cost = loss(out,pgd_target)
        # backward pass
        cost.backward()
        # get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        # convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0][1] - bounds[0][0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1][1] - bounds[1][0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2][1] - bounds[2][0])) * per[:,2,:,:]
        # descent
        adv = img.data + per.cuda()
        # clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0][0],bounds[0][1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1][0],bounds[1][1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2][0],bounds[2][1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
    img = data + noise
    return img

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    #losses = AverageMeter('Loss', ':.4e')
    celoss = AverageMeter('CELoss', ':.4e')
    regloss = AverageMeter('RegLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, celoss, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        
        images = images.cuda()
        B,C,H,W = images.size()
        target = target.cuda(args.gpu, non_blocking=True)
        
        N = torch.sign(torch.tensor([0.5]).cuda() - torch.rand_like(images).cuda()).cuda()
        adv_data = images + (args.Bval/255.0)*N
        adv_data = torch.clamp(adv_data,0.0,1.0)
        
        model.eval()
        
        alt = (i%2)
        adv_data = Guided_Attack(args,model,criterion,adv_data,target,eps=args.Feps/255.0,steps=1,data=images.detach(),l2_reg=args.l2_reg,alt=alt,B=B)
        
        delta = adv_data - images
        delta = torch.clamp(delta,-8.0/255.0,8.0/255)
        adv_data = images+delta
        adv_data = torch.clamp(adv_data,0.0,1.0)
        
        model.train()
        adv_out  = model(adv_data)
        out  = model(images)
        
        Q_out = nn.Softmax(dim=1)(adv_out)
        P_out = nn.Softmax(dim=1)(out)
        
        '''LOSS COMPUTATION'''
        
        closs = criterion(out,target)        
        reg_loss =  ((P_out - Q_out)**2.0).sum(1).mean(0)
        
        loss = args.lce*closs + args.l2_reg*reg_loss   
        # measure accuracy and record loss
        acc1, acc5 = accuracy(out, target, topk=(1, 5))
        
        celoss.update(closs.item(), images.size(0))
        regloss.update(reg_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Val Set')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def pgdvalidate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('PGD'+str(steps)+'Acc@1', ':6.2f')
    top5 = AverageMeter('PGD'+str(steps)+'Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Val Set: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute PGD attack
        data = pgd(args,model,criterion,images,target,eps,eps_iter,bounds=[[0,1],[0,1],[0,1]],steps=steps)
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(' * Epoch: '+str(args.start_epoch)+' eps: '+str(eps)+' eps_iter: '+str(eps_iter)+' steps: '+str(steps))
    
    print(' * PGD'+str(steps)+'Acc@1 {top1.avg:.3f}  PGDAcc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
          

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    
    if epoch in [60,90]:
        args.lr = args.lr/10.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            
    if epoch == 90:
        args.l2_reg = args.l2_reg*args.mul
    
    print('lce: '+str(args.lce)+' l2reg: '+str(args.l2_reg)+' mul: '+str(args.mul))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
