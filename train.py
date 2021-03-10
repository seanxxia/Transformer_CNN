import os
import time
import math 
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import torch.nn.functional as F

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from dataset.Tusimple import Tusimple
from dataset.CULane import CULane
from model.model import *

# ------------ config ------------
device = torch.device('cuda:0')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ------------ train data ------------
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_seg = transforms.Compose([
    transforms.Resize((224, 224), 0),
    transforms.ToTensor()
])

data_transforms = [transform_img, transform_seg]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)
train_dataset = Tusimple('/home/aimmlab/Xia/tusimple_ultra/', image_set='train', transforms=data_transforms, augmentation=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate, num_workers=8)

valid_dataset = Tusimple('/home/aimmlab/Xia/tusimple_ultra/', image_set='val', transforms=data_transforms)
valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=valid_dataset.collate, num_workers=8)

# train_dataset = CULane('/home/aimmlab/Xia/Transformer/culane', image_set='train', transforms=data_transforms)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate, num_workers=8)

# valid_dataset = CULane('/home/aimmlab/Xia/Transformer/culane', image_set='val', transforms=data_transforms)
# valid_loader = DataLoader(valid_dataset, batch_size=64, collate_fn=valid_dataset.collate, num_workers=8)

def train(net, optimizer):
    net.train()
    print(" | Learning Rate: {}".format(optimizer.param_groups[0]['lr']))

    train_loss = 0.

    progressbar = tqdm(range(len(train_loader)))
    for batch_idx, (img, seg, _) in enumerate(train_loader):
        img = img.to(device)
        seg = seg.to(device)
        optimizer.zero_grad()
        seg_pred = net(img)

        loss = seg_loss(seg_pred, seg)
        loss.backward()

        train_loss += loss.item()

        optimizer.step()
        progressbar.update(1)

    train_loss /= len(train_loader)
    progressbar.close()

    print(" | Epoch Loss: {}".format(train_loss))
    return train_loss

def valid(net):
    net.eval()
    
    valid_loss = 0.

    progressbar = tqdm(range(len(valid_loader)))
    with torch.no_grad():
        for batch_idx, (img, seg, _) in enumerate(valid_loader):
            img = img.to(device)
            seg = seg.to(device)

            seg_pred = net(img)
            loss = seg_loss(seg_pred, seg)
            
            valid_loss += loss.item()
            progressbar.update(1)

    valid_loss /= len(valid_loader)
    progressbar.close()

    print(" | Epoch Loss: {}".format(valid_loss))

    return valid_loss

if __name__ == "__main__":
    net = SCNN_ViT(train=True)
    net = net.to(device)

    seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1, 1, 1, 1]))
    seg_loss = seg_loss.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.03)
    # warm_up_with_multistep_lr = lambda epoch: epoch / 5 if epoch <= 5 else 0.1**len([m for m in [30, 45] if m <= epoch])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    warm_up_with_cosine_lr = lambda epoch: epoch / 5 if epoch <= 5 else 0.5 * ( math.cos((epoch - 5) /(50 - 5) * math.pi) + 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    best_val = 2e50
    best_epoch = 0
    
    print('---------------------------------------------------------------------------------------------\n')
    start = time.time()
    print(' | Start Training')
    for epoch in range(1, 201):
        print(" | Train Epoch: {}".format(epoch))
        t_loss = train(net, optimizer)

        print('\n---------------------------------------------------------------------------------------------\n')
        print(" | Valid Epoch: {}".format(epoch))
        v_loss = valid(net)
        if epoch%10 == 0:
            torch.save(net.state_dict(), '/home/aimmlab/Xia/Transformer_CNN/experiments/tusimple/segmentation/SCNN/' + 'epoch_{}.pth'.format(epoch))

        if v_loss < best_val:
            print(" | Update Best Model")
            best_val = v_loss
            best_epoch = epoch
        
        print('\n---------------------------------------------------------------------------------------------\n')
        scheduler.step()
    
    print(" | Total Training Time: {} seconds".format(time.time() - start))
    print(" | Best Epoch: Epoch {}".format(best_epoch))
    print(" | Best Epoch Loss: {}".format(best_val))
