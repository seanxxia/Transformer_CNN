import argparse
import json
import os
import shutil
import time
import warnings
import math

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F

from dataset.Tusimple import Tusimple
from model.model import *

from utils.prob2lines import getLane

# ------------ config ------------
exp_dir = './experiments/tusimple/segmentation'
device = torch.device('cuda:0')

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

test_dataset = Tusimple('/home/aimmlab/Xia/tusimple_ultra/', image_set='test', transforms=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=test_dataset.collate, num_workers=8)

def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders

net = SCNN_ViT()
save_name = './experiments/tusimple/segmentation/SCNN/epoch_90.pth'
net.load_state_dict(torch.load(save_name, map_location='cpu'))
net = net.to(device)
net.eval()

# ------------ test ------------
out_path = os.path.join(exp_dir, "coord_output")
evaluation_path = os.path.join(exp_dir, "evaluate")
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(evaluation_path):
    os.mkdir(evaluation_path)
dump_to_json = []

progressbar = tqdm(range(len(test_loader)))
with torch.no_grad():
    for batch_idx, (imgs, _, img_name) in enumerate(test_loader):
        imgs = imgs.to(device)

        seg_pred = net(imgs)
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()

        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 for i in range(4)]
            lane_coords = getLane.prob2lines_tusimple(seg, exist, smooth=False, thresh=0.5, resize_shape=(720, 1280), y_px_gap=10, pts=56)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])

            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(*path_tree[-4:])
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            dump_to_json.append(json.dumps(json_dict))

        progressbar.update(1)
    progressbar.close()

with open(os.path.join(out_path, "predict_test.json"), "w") as f:
    for line in dump_to_json:
        print(line, end="\n", file=f)

# ---- evaluate ----
from utils.lane_evaluation.tusimple.lane import LaneEval

eval_result = LaneEval.bench_one_submit(os.path.join(out_path, "predict_test.json"),
                                        os.path.join('/home/aimmlab/Xia/tusimple_ultra/', 'test_label.json'))
print(eval_result)
with open(os.path.join(evaluation_path, "evaluation_result.txt"), "w") as f:
    print(eval_result, file=f)
