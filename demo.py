import argparse
import json
import os
import shutil
import time
import warnings
import math
import cv2
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

# from dataset import Tusimple
from model.model import *

# ------------ config ------------
device = torch.device('cpu')

# ------------ train data ------------

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if __name__ == "__main__":
    net = ViT()
    net.load_state_dict(torch.load('./experiments/tusimple/segmentation/epoch_50.pth', map_location='cpu'))
    net.to(device)
    net.eval()
    color = [(255, 125, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    
    jpg = sys.argv[1]

    with torch.no_grad():        
        img = Image.open(jpg)
        orig = cv2.imread(jpg)
        img = transform_img(img)
        img = img.to(device)
        img = img.unsqueeze(0)

        for _ in range(100):
            _ = net(img)

        t_all = []
        for i in range(100):
            t1 = time.time()
            seg_pred = net(img)
            t2 = time.time()
            t_all.append(t2 - t1)
            seg_pred = seg_pred.squeeze(0).cpu().numpy()
            seg_pred = seg_pred.transpose(1, 2, 0)
            # seg_pred = cv2.resize(seg_pred, (1640, 590), cv2.INTER_NEAREST)
            seg_pred = cv2.resize(seg_pred, (1280, 720), cv2.INTER_NEAREST)
            coord_mask = np.argmax(seg_pred, axis=2)

            for i in range(0, 4):
                mask = coord_mask == (i + 1)
                pred_i = seg_pred[:, :, i+1] * mask
                # for j in range(12, 29):
                for j in range(16, 72):
                    row = pred_i[10*j]
                    # row = pred_i[20*j]
                    m = np.max(row)
                    p = np.argmax(row)
                    if m > 0:
                        cv2.circle(orig, (p, 10*j), 5, color[i], -1)
                        # cv2.circle(orig, (p, 20*j), 5, color[i], -1)

            cv2.imwrite('result.jpg', orig)

        print('\tAverage FPS: ', 1 / np.mean(t_all))
        print('\tFastest FPS: ', 1 / np.min(t_all))
        print('\tSlowest FPS: ', 1 / np.max(t_all))
