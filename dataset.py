import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from helpers import read_labels, read_data_names

####아래 튜토리얼과 매우 비슷함!!
# #https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class CoordDataset(Dataset):
    def __init__(self, data_location, coords, data_names, transform_list = None):
        super(CoordDataset).__init__()
        self.data_location = data_location
        self.labels = coords
        self.data_names = data_names
        self.size = len(data_names)
        self.transform_list = transform_list
        self.toTensor = transforms.ToTensor()
        self.nor = transforms.Normalize((0.5,), (0.5,))
        self.toImage = transforms.ToPILImage()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_location, self.data_names[idx])
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform_list != None:
            for transform in self.transform_list:
                image, label = transform(image, label)

        image = self.toTensor(image)
        return {'image' : image, 'label' : label}


class SpineDataset(Dataset):
    def __init__(self, data_location, label_location, data_names, coords_rel, transform = None, is_segment = True):
        super(SpineDataset).__init__()
        self.data_location = data_location
        self.label_location = label_location
        self.coords_rel = coords_rel
        self.data_names = data_names
        self.size = len(data_names)
        self.transform = transform
        self.is_segment = is_segment

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_location, self.data_names[idx])
        label_path = os.path.join(self.label_location, self.data_names[idx] + '.npy')

        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
            ##특정 트랜스폼은 lab에 적용 X!
            #lab = self.transform(lab)
        else:
            trs = transforms.ToTensor()
            img = trs(img)

        if self.is_segment:
            lab = np.load(label_path)
            lab = lab.reshape((1,lab.shape[0], lab.shape[1]))
            sample = {'image' : img, 'label' : lab}
        else:
            coord = self.coords_rel[idx]
            sample = {'image' : img, 'label' : coord}

        return sample
