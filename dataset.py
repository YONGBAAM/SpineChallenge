import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from label_io import read_labels, read_data_names, plot_image
from label_transform import CoordCustomPad, CoordHorizontalFlip,CoordRandomRotate, CoordLabelNormalize, CoordResize, CoordVerticalFlip

###############################################
#
#       Refactoring Finished
#
###############################################

RHP = [
        CoordRandomRotate(max_angle=10, expand=False),
        CoordHorizontalFlip(0.5),
        CoordCustomPad(512 / 256),
        CoordResize((512, 256)),
        CoordLabelNormalize()
]
NOPAD = [
    CoordRandomRotate(max_angle=10, expand=False),
    CoordHorizontalFlip(0.5),
    CoordResize((512, 256)),
    CoordLabelNormalize()
]
UNF = [
    CoordRandomRotate(max_angle=10, expand=False),
    CoordHorizontalFlip(0.5),
    CoordCustomPad(512 / 256, random='uniform'),
    CoordResize((512, 256)),
    CoordLabelNormalize()
]
UNF_E = [
    CoordRandomRotate(max_angle=10, expand=True),
    CoordHorizontalFlip(0.5),
    CoordCustomPad(512 / 256, random='uniform'),
    CoordResize((512, 256)),
    CoordLabelNormalize()
]
NOPAD_VAL = [
    #    CoordCustomPad(512 / 256),
    CoordResize((512, 256)),
    CoordLabelNormalize()
]

PAD_VAL = [
    CoordCustomPad(512 / 256),
    CoordResize((512, 256)),
    CoordLabelNormalize()
]

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
        image = self.nor(image)
        return {'image' : image, 'label' : label}

def get_loader_train_val(tfm_train = 'nopad', tfm_val = 'nopad', batch_size_tr=64, batch_size_val=1, shuffle = True):
    tfm = tfm_train
    if type(tfm) == type('PAD'):
        if tfm.lower() == 'pad_val':
            tfm = PAD_VAL
        elif tfm.lower() == 'nopad_val':
            tfm = NOPAD_VAL
        elif tfm.lower() == 'pad' or tfm.lower() == 'unf' or tfm.lower() == 'uniform':
            tfm = UNF
        elif tfm.lower() == 'nopad':
            tfm = NOPAD
        elif tfm.lower() == 'rhp':
            tfm = RHP
        else:
            tfm = None
    tfm_train = tfm

    tfm = tfm_val
    if type(tfm) == type('PAD'):
        if tfm.lower() == 'pad_val' or tfm.lower() == 'pad':
            tfm = PAD_VAL
        elif tfm.lower() == 'nopad_val' or tfm.lower() == 'nopad':
            tfm = NOPAD_VAL
        else:
            tfm = None
    tfm_val = tfm


    data_path = './train_images'
    label_path = './train_labels'

    val_ratio = 0.1

    labels = read_labels(label_path)
    data_names = read_data_names(label_path)

    N_all = len(data_names)
    N_val = int(N_all * val_ratio)
    N_train = N_all - N_val
    # get train and validation data set
    data_names_train = []
    data_names_val = []
    labels_train = []
    labels_val = []

    if not os.path.exists(os.path.join('./', 'val_permutation.npy')):
        print('reset permutation')
        permutation = np.random.permutation(N_all)
        np.save(os.path.join('./', 'val_permutation.npy'), permutation)
    else:
        permutation = np.load(os.path.join('./', 'val_permutation.npy'))

    for ind in permutation[:N_train]:
        data_names_train.append(data_names[ind])
        labels_train.append(labels[ind])
    labels_train = np.asarray(labels_train)

    for ind in permutation[N_train:]:
        data_names_val.append(data_names[ind])
        labels_val.append(labels[ind])
    labels_val = np.asarray(labels_val)
    #########################
    dset_train = CoordDataset(data_path, labels_train, data_names_train, transform_list=tfm_train)
    dset_val = CoordDataset(data_path, labels_val, data_names_val, transform_list=tfm_val)

    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size_tr, shuffle=shuffle)
    loader_val = DataLoader(dataset=dset_val, batch_size=batch_size_val, shuffle=False)
    return loader_train ,loader_val

def get_loader_train(tfm = 'nopad', batch_size = 64, shuffle = False):
    if type(tfm) == type('PAD'):
        if tfm.lower() == 'pad_val':
            tfm = PAD_VAL
        elif tfm.lower() == 'nopad_val':
            tfm = NOPAD_VAL
        elif tfm.lower() == 'pad' or tfm.lower() == 'unf' or tfm.lower() == 'uniform':
            tfm = UNF
        elif tfm.lower() == 'nopad':
            tfm = NOPAD
        elif tfm.lower() == 'rhp':
            tfm = RHP
        else:
            tfm = None

    data_path = './train_images'
    label_path = './train_labels'
    labels = read_labels(label_path)
    data_names = read_data_names(label_path)
    dset_train = CoordDataset(data_path, labels, data_names, transform_list=tfm)

    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=shuffle)
    return loader_train

def get_loader_test(tfm = 'nopad', batch_size = 1, shuffle = False):
    if type(tfm) == type('PAD'):
        if tfm.lower() == 'pad_val' or tfm.lower() == 'pad':
            tfm = PAD_VAL
        elif tfm.lower() == 'nopad_val' or tfm.lower() == 'nopad':
            tfm = NOPAD_VAL
        else:
            tfm = None


    data_path = './test_images'
    label_path = './test_labels'
    labels = read_labels(label_path)
    data_names = read_data_names(label_path)
    dset_test = CoordDataset(data_path, labels, data_names, transform_list=tfm)

    loader_test = DataLoader(dataset=dset_test, batch_size=batch_size, shuffle=shuffle)
    return loader_test

if __name__ == '__main__':
    ############Dataset, transform test


    RH_E = [
        CoordRandomRotate(max_angle=10, expand=True),
        CoordHorizontalFlip(0.5),
        CoordCustomPad(512 / 256),
        CoordResize((512, 256)),
        CoordLabelNormalize()
    ]

    ###############################
    #   Get target loader
    tloader = get_loader_train(tfm = RH_E, batch_size=64, shuffle = False)


    index = 0
    for testdata in tloader:
        imgs = testdata['image'].cpu().detach().numpy()
        labs = testdata['label'].cpu().detach().numpy()
        for i, img in enumerate(imgs):
            lab = labs[i]
            plt.figure()
            plot_image(img, coord_red= lab)
            plt.title('train {}'.format(index))
            plt.show()


#
# class SpineDataset(Dataset):
#     def __init__(self, data_location, label_location, data_names, coords_rel, transform = None, is_segment = True):
#         super(SpineDataset).__init__()
#         self.data_location = data_location
#         self.label_location = label_location
#         self.coords_rel = coords_rel
#         self.data_names = data_names
#         self.size = len(data_names)
#         self.transform = transform
#         self.is_segment = is_segment
#
#     def __len__(self):
#         return self.size
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.data_location, self.data_names[idx])
#         label_path = os.path.join(self.label_location, self.data_names[idx] + '.npy')
#
#         img = Image.open(img_path)
#         if self.transform is not None:
#             img = self.transform(img)
#             ##특정 트랜스폼은 lab에 적용 X!
#             #lab = self.transform(lab)
#         else:
#             trs = transforms.ToTensor()
#             img = trs(img)
#
#         if self.is_segment:
#             lab = np.load(label_path)
#             lab = lab.reshape((1,lab.shape[0], lab.shape[1]))
#             sample = {'image' : img, 'label' : lab}
#         else:
#             coord = self.coords_rel[idx]
#             sample = {'image' : img, 'label' : coord}
#
#         return sample
