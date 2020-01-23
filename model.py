import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CCB(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CCB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= input_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.bn1 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x=  F.relu(self.bn1(self.conv2(x)))
        return x

class CCU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CCU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= input_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.upconv1 = nn.Upsample(scale_factor=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upconv1(x)
        return x

class SegmentNet(nn.Module):
    def __init__(self):
        super(SegmentNet, self).__init__()
        self.ccb1 = CCB(1,16)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.ccb2 = CCB(16,32)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.ccb3 = CCB(32,64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.ccb4 = CCB(64,64)

        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.ccb5 = CCB(64,64)
        self.ups = nn.Upsample(scale_factor=2)##나온거 테스트 보고 하기

        self.ccu1 = CCU(128,16)
        self.ccu2 = CCU(80,16)
        self.ccu3 = CCU(48,16)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding = 1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        #Testing for size
        y1 = self.ccb1(x) # 508 252

        y2 = self.pool1(y1)
        y2 = self.ccb2(y2) #250 122

        y3 = self.pool2(y2)
        y3 = self.ccb3(y3) #121 57

        y4 = self.pool3(y3)
        y4 = self.ccb4(y4) #56 24

        y5 = self.pool4(y4)
        y5 = self.ccb5(y5) #chw 64 24 8

        y5 = self.ups(y5)
        y44 = torch.cat((y5,y4), dim = 1)
        y44 = self.ccu1(y44)

        y33 = torch.cat((y44, y3), dim = 1)
        y33 = self.ccu2(y33)

        y22 = torch.cat((y33, y2), dim = 1)
        y22 = self.ccu3(y22)

        y11 = torch.cat((y22, y1), dim = 1)
        y11 = self.conv3(y11)
        y11 = self.conv1(y11)

        return y11

def get_dependency_matrix(all_vertex = 68):
    S = torch.zeros(all_vertex, all_vertex)

    for i in range(1, int(all_vertex/2) -1): #1,....32
        S[2*i][2*i+1] = 1
        S[2*i][2*i+2] = 1
        S[2*i+1][2*i-1] = 1
        S[2*i+1][2*i] = 1
        S[2*i+1][2*i+3] = 1
        S[2*i][2*i-2] = 1
    i = 0
    S[2 * i][2 * i + 1] = 1
    S[2 * i][2 * i + 2] = 1
    S[2 * i + 1][2 * i] = 1
    S[2 * i + 1][2 * i + 3] = 1
    i = int(all_vertex/2) -1
    S[2 * i][2 * i + 1] = 1
    S[2 * i + 1][2 * i - 1] = 1
    S[2 * i + 1][2 * i] = 1
    S[2 * i][2 * i - 2] = 1
    return S

class SpinalStructured(nn.Module):
    def __init__(self, output_dim = 68):
        super(SpinalStructured, self).__init__()
        self.S = nn.Parameter(get_dependency_matrix(output_dim + 4))
        #S 걍 Linear로??
        self.S.requires_grad = False

    def forward(self,x):
        x = x.reshape(x.shape[0], -1, 2) #N 68+4 2
        x = x.transpose(1,2)
        x = torch.einsum("abc,cd->abd", (x, self.S))
        x = x.transpose(2,1)
        x = x[:,2:-2,:]
        x = x.reshape(x.shape[0], -1)
        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.S = self.S.to(*args, **kwargs)
        return self

def get_feature_extractor(requires_grad = False):
    resnet_based = torchvision.models.resnet101(pretrained=True)
    for param in resnet_based.parameters():
        param.requires_grad = requires_grad
    fe = nn.Sequential(*list(resnet_based.children())[:-1])
    return fe



def get_classifier():
    classifier = nn.Sequential(
        nn.Flatten(),#2048 for resnet 101
        nn.Linear(2048,2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 136 + 8),
        SpinalStructured(output_dim=68)
    )
    return classifier

class LandmarkNet(nn.Module):
    def __init__(self, classifier = None, requires_grad = False):
        super(LandmarkNet, self).__init__()
        self.extractor = get_feature_extractor(requires_grad=requires_grad)
        if classifier == None:
            self.classifier = get_classifier()
        else:
            self.classifier = classifier

    def forward(self, x):
        if x.shape[1] ==1:#grayscale image
            x = x.repeat((1,3,1,1))
        return self.classifier(self.extractor(x))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        self.extractor = self.extractor.to(*args, **kwargs)
        return self


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import os
    import scipy.io as spio
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    from model import BoostLayer
    import torch
    import torch.nn as nn
    from model import SpineNet
    from dataset import SpineDataset
    import torchvision.transforms as transforms

    data_path = './resized_images'
    label_path = './resized_labels'
    save_path = './model'

    val_ratio = 0.1
    batch_size = 2

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    from helpers import read_data_names, read_labels
    ##Get DataLoader
    # get label and dataname list
    labels = read_labels(label_path)
    data_names = read_data_names(label_path)

    N_all = len(data_names)
    N_val = int(N_all * val_ratio)
    N_train = N_all - N_val

    # get train and validation data set
    data_names_train = []
    data_names_val = []
    label_train = []  # np.zeros(N_train, label.shape[1])
    label_val = []  # np.zeros(N_val, label.shape[1])

    if True:
        permutation = np.random.permutation(N_all)
        for ind in permutation[:N_train]:
            data_names_train.append(data_names[ind])
            label_train.append(labels[ind])
        for ind in permutation[N_train:]:
            data_names_val.append(data_names[ind])
            label_val.append(labels[ind])

        label_val = np.asarray(label_val)
        label_train = np.asarray(label_train)

    dset_train = SpineDataset(data_path, label_train, data_names=data_names_train, transform=transform)
    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True)

    _train_data = next(iter(loader_train))
    _imgs = _train_data['image']
    _labels = _train_data['label']

    model = SegmentNet()

    ys = model(_imgs)

    for y in ys:
        print('{}'.format(y[0].shape))


#class BoostLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, threshold = 2, device = None):
#         super(BoostLayer, self).__init__()
#         self.n_sigma = threshold
#
#
#         self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
#         nn.init.xavier_normal_(self.weight)
#         self.bias1 = nn.Parameter(torch.zeros(output_dim))
#         self.bias2 = nn.Parameter(torch.zeros(input_dim))
#
#         #이거 좀더 팬시하게 할 수 있을텐데
#         if device is not None:
#             self.weight.to(device)
#             self.bias1.to(device)
#             self.bias2.to(device)
#
#     def forward(self, x):
#         N, C = x.shape[0], x.shape[1]
#         R = x.mm(self.weight) + self.bias1
#         R = F.relu(R).mm(self.weight.transpose(0,1))
#         errors = (x-R)**2
#         mean = torch.mean(x, dim = -1, keepdim = False)
#         std = torch.std(x, dim = -1, keepdim = False)
#         threshold = (self.n_sigma * std) ** 2
#         for n in range(N):
#             mask = errors[n]>threshold[n].item()
#             x[n][mask] = mean[n]
#         y = x.mm(self.weight) + self.bias1
#         return y
#
#     def to(self, *args, **kwargs):
#         self = super().to(*args, **kwargs)
#         self.weight = self.weight.to(*args, **kwargs)
#         self.bias1 = self.bias1.to(*args, **kwargs)
#         self.bias2 = self.bias2.to(*args, **kwargs)
#         return self