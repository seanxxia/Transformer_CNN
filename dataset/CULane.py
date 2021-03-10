import cv2
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CULane(Dataset):
    def __init__(self, path, image_set, task='instance', transforms=None):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.task = task

        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()


    def createIndex(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}_gt.txt".format(self.image_set))

        self.img_list = []
        self.seg_list = []
        self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.seg_list.append(os.path.join(self.data_dir_path, l[1][1:]))
                self.exist_list.append([int(x) for x in l[2:]])

    def createIndex_test(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))

        self.img_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, line[1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = self.transforms[0](img)
        name = self.img_list[idx]

        if self.image_set != 'test':
            seg = Image.open(self.seg_list[idx])
            seg = self.transforms[1](seg)[0] * 255
            exist = torch.tensor(self.exist_list[idx])
        else:
            seg = None
            exist = None

        return img, seg, exist, name

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0][0], torch.Tensor):
            img = torch.stack([b[0] for b in batch])
        else:
            img = [b[0] for b in batch]

        if batch[0][1] is None:
            seg = None
            exist = None
        elif isinstance(batch[0][1], torch.Tensor):
            seg = torch.stack([b[1] for b in batch])
            exist = torch.stack([b[2] for b in batch])
        else:
            seg = [b[1] for b in batch]
            exist = [b[2] for b in batch]

        name = [b[3] for b in batch]

        return img, seg, exist, name