import json
import os
import math
import random

from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class Tusimple(Dataset):
    
    TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
    VAL_SET = ['label_data_0531.json']
    TEST_SET = ['test_label.json']

    def __init__(self, path, image_set, transforms=None, augmentation=False):
        super(Tusimple, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.augmentation = augmentation
        self.label = 'segLabel'

        if not os.path.exists(os.path.join(path, self.label)):
            print("Label is going to get generated into dir: {} ...".format(os.path.join(path, self.label)))
            self.generate_label()
        self.createIndex()

    def createIndex(self):
        self.img_list = []
        self.seg_list = []

        listfile = os.path.join(self.data_dir_path, self.label, "list", "{}_gt.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))
                self.seg_list.append(os.path.join(self.data_dir_path, l[1][1:]))

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = self.transforms[0](img)
        name = self.img_list[idx]

        if self.image_set != 'test':
            seg = Image.open(self.seg_list[idx])
            seg = self.transforms[1](seg)[0] * 255
            seg = seg.long()
            
            if self.augmentation:
                flag = random.random() < 0.3
                if flag:
                    img = torch.flip(img, dims=[2])
                    seg = torch.flip(seg, dims=[1])

        else:
            seg = None

        return img, seg, name

    def __len__(self):
        return len(self.img_list)

    def generate_label(self):
        save_dir = os.path.join(self.data_dir_path, self.label)
        os.makedirs(save_dir, exist_ok=True)

        # --------- merge json into one file ---------
        with open(os.path.join(save_dir, "train.json"), "w") as outfile:
            for json_name in self.TRAIN_SET:
                with open(os.path.join(self.data_dir_path, json_name)) as infile:
                    for line in infile:
                        outfile.write(line)

        with open(os.path.join(save_dir, "val.json"), "w") as outfile:
            for json_name in self.VAL_SET:
                with open(os.path.join(self.data_dir_path, json_name)) as infile:
                    for line in infile:
                        outfile.write(line)

        with open(os.path.join(save_dir, "test.json"), "w") as outfile:
            for json_name in self.TEST_SET:
                with open(os.path.join(self.data_dir_path, json_name)) as infile:
                    for line in infile:
                        outfile.write(line)

        self._gen_label_for_json('train')
        print("train set is done")
        self._gen_label_for_json('val')
        print("val set is done")
        self._gen_label_for_json('test')
        print("test set is done")

    def _gen_label_for_json(self, image_set):
        H, W = 720, 1280
        SEG_WIDTH = 30
        save_dir = self.label

        os.makedirs(os.path.join(self.data_dir_path, save_dir, "list"), exist_ok=True)
        list_f = open(os.path.join(self.data_dir_path, save_dir, "list", "{}_gt.txt".format(image_set)), "w")

        json_path = os.path.join(self.data_dir_path, save_dir, "{}.json".format(image_set))
        with open(json_path) as f:
            for line in f:
                label = json.loads(line)

                # ---------- clean and sort lanes -------------
                lanes = []
                _lanes = []
                slope = [] # identify 1st, 2nd, 3rd, 4th lane through slope
                for i in range(len(label['lanes'])):
                    l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                    if (len(l)>1):
                        _lanes.append(l)
                        slope.append(np.arctan2(l[-1][1]-l[0][1], l[0][0]-l[-1][0]) / np.pi * 180)
                _lanes = [_lanes[i] for i in np.argsort(slope)]
                slope = [slope[i] for i in np.argsort(slope)]

                idx_1 = None
                idx_2 = None
                idx_3 = None
                idx_4 = None
                for i in range(len(slope)):
                    if slope[i]<=90:
                        idx_2 = i
                        idx_1 = i-1 if i>0 else None
                    else:
                        idx_3 = i
                        idx_4 = i+1 if i+1 < len(slope) else None
                        break
                lanes.append([] if idx_1 is None else _lanes[idx_1])
                lanes.append([] if idx_2 is None else _lanes[idx_2])
                lanes.append([] if idx_3 is None else _lanes[idx_3])
                lanes.append([] if idx_4 is None else _lanes[idx_4])
                # ---------------------------------------------

                img_path = label['raw_file']
                seg_img = np.zeros((H, W, 3))
                list_str = []
                for i in range(len(lanes)):
                    coords = lanes[i]
                    if len(coords) < 4:
                        continue
                    for j in range(len(coords)-1):
                        cv2.line(seg_img, coords[j], coords[j+1], (i+1, i+1, i+1), SEG_WIDTH//2)

                label_path = img_path.split("/")
                label_path, img_name = os.path.join(self.data_dir_path, save_dir, label_path[1], label_path[2]), label_path[3]
                os.makedirs(label_path, exist_ok=True)
                seg_path = os.path.join(label_path, img_name[:-3]+"png")
                cv2.imwrite(seg_path, seg_img)

                seg_path = "/".join([save_dir, *img_path.split("/")[1:3], img_name[:-3]+"png"])

                if seg_path[0] != '/':
                    seg_path = '/' + seg_path
                if img_path[0] != '/':
                    img_path = '/' + img_path
                
                list_str.insert(0, seg_path)
                list_str.insert(0, img_path)
                list_str = " ".join(list_str) + "\n"
                list_f.write(list_str)

        list_f.close()

    @staticmethod
    def collate(batch):
        if isinstance(batch[0][0], torch.Tensor):
            img = torch.stack([b[0] for b in batch])
        else:
            img = [b[0] for b in batch]

        if batch[0][1] is None:
            seg = None
        elif isinstance(batch[0][1], torch.Tensor):
            seg = torch.stack([b[1] for b in batch])
        else:
            seg = [b[1] for b in batch]

        name = [b[2] for b in batch]
        return img, seg, name
