
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
import sys
import os
import warnings
from model import CSRNet
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
import dataset
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def setup(rank, world_size):
    """Setups the environment"""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    """Kills the processes"""

    dist.destroy_process_group()

def train(args,  model, train_loader, criterion, optimizer, epoch, rank):
    """Trains the model"""

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    if rank == 0 :
        print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.to(rank)
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).to(rank)
        target = Variable(target)

        loss = criterion(output, target) 
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(model, val_loader, rank):
    """Validates the training"""

    if rank == 0:
        print ('begin test')
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(val_loader):
        img = img.to(rank)
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(rank))
        
    mae = mae/len(val_loader)

    if rank == 0:    
        print(' * MAE {mae:.3f} '
                .format(mae=mae))

    return mae    

def run(rank, world_size, args, use_cuda):
    """Runs the DDP training code on multi gpus"""

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)

    global best_prec1

    best_prec1 = 1e6

    #--------------- Setup -------------#
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    #-----------------------------------#
   
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = CSRNet().to(rank)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    if args.pre:
        if os.path.isfile(args.pre):
            if rank == 0:
                print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre, map_location=torch.device(rank))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(torch.load(args.pre, map_location=torch.device(rank)), strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0: 
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.pre, checkpoint['epoch']))
        else:
            if rank == 0:
                print("=> no checkpoint found at '{}'".format(args.pre))

    criterion = nn.MSELoss(size_average=False).to(rank)

    model = DDP(model, device_ids=[rank])

    
    
    train_dataset = dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True,
                       batch_size=args.batch_size,
                       num_workers=args.workers)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        pin_memory=True,
        shuffle=False,
        num_workers=args.workers,
        batch_size=args.batch_size)

    val_dataset = dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler)
    

    

    for epoch in range(args.start_epoch, args.epochs):   
        adjust_learning_rate(optimizer, epoch, args)
        train(args, model, train_loader, criterion, optimizer, epoch, rank)
        prec1 = validate(model, val_loader, rank)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        if rank == 0:
            print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
            
            save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),}, is_best,str(args.task))
    cleanup()

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def main():
    """Sets the training parameters"""

    parser = argparse.ArgumentParser(description='PyTorch CSRNet')
    
    
    parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
    parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')
    parser.add_argument('-gpus', metavar='gpus', default=2, type=int,
                        help='Number of GPUs')
    parser.add_argument('-epochs', metavar='EPOCHS', default=400,
                    help='number of epochs')
    parser.add_argument('-start_epoch', metavar='EPOCHS', default=0,
                    help='number of epochs')
    parser.add_argument('-pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')
    parser.add_argument('-no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-task',metavar='TASK',default=0, type=str,
                    help='task id to use.')
    parser.add_argument('-batch_size',metavar='BATCH', default=1, type=int,
                    help='batch_size')
    parser.add_argument('-workers',metavar='WORKRS', default=4 , type=int,
                    help='workers')
    parser.add_argument('-lr',metavar='LR', default=1e-7 , type=float,
                    help='lr')
    parser.add_argument('-momentum',metavar='MOMENTUM', default=0.95 , type=float,
                    help='momentum')
    parser.add_argument('-decay',metavar='DECAY', default=5*1e-4 , type=float,
                    help='decay')
    parser.add_argument('-seed',metavar='SEED', default=time.time() , type=float,
                    help='seed')
    parser.add_argument('-print_freq',metavar='PRINT_FREQ', default=30 , type=int,
                    help='print_freq')
    parser.add_argument('-original_lr',metavar='ORIGINAL_LR', default=1e-7 , type=float,
                    help='original_lr')
    parser.add_argument('-scales',metavar='SCALES', default=[1,1,1,1] , type=list,
                    help='scales')
    parser.add_argument('-steps',metavar='STEPS', default=[-1,1,100,150] , type=list,
                    help='steps')
                
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    world_size = args.gpus    
    
    if torch.cuda.device_count() > 1:
      print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")

    #########################################################
    mp.spawn(run, args=(world_size, args, use_cuda), nprocs=world_size, join=True)    
    #########################################################


if __name__ == '__main__':
    main()
