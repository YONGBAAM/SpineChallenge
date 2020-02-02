import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

################################
#
#       Refactoring Finished
#
################################



def get_feature_extractor(resnet_dim = 101, requires_grad = False, PoolDrop = True):
    if resnet_dim == 34:#512 16 8
        resnet_based = torchvision.models.resnet34(pretrained=True)
    elif resnet_dim == 18: #512 16 8
        resnet_based = torchvision.models.resnet18(pretrained=True)
    else:#2048 16 8
        resnet_based = torchvision.models.resnet101(pretrained=True)

    for param in resnet_based.parameters():
        param.requires_grad = requires_grad

    if PoolDrop:
        fe = nn.Sequential(*list(resnet_based.children())[:-2])
    else:
        fe = nn.Sequential(*list(resnet_based.children())[:-1])
        print('Resnet does not drop pool layer')

    return fe

def get_classifier_deep(dropout = 0.5):
    lst = [
        nn.Conv2d(2048, 512, kernel_size=(1, 1)),  # 16 8
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(512, 512, 3, padding=1),  # 16 8
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(512, 128, kernel_size=(1, 1)),  # 16 8
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),  #16*8

        #nn.AdaptiveAvgPool2d(output_size=(8, 4)),
        nn.Conv2d(128,128,kernel_size=(4,4), stride = 2, padding=1),  #8 * 4
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Flatten(),
        nn.Linear(8*4*128, 1024),
        nn.ReLU(),
        nn.Dropout(dropout),

        nn.Linear(1024,136)

        ]
    classifier = nn.Sequential(*lst)
    return classifier

def get_classifier_conv(dropout = 0.5, with_spine = False):
    lst = [nn.Conv2d(2048,512,kernel_size=(1,1)), #16 8
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512,512,3, padding = 1), #16 8
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512,512,3,padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(), #128*16*8
        nn.AdaptiveAvgPool2d(output_size = (2,2)),
        nn.Flatten(),
        nn.Linear(512*2*2,512),
        nn.ReLU(),
        nn.Dropout(dropout)
    ]
    if with_spine:
        print("Do not Use it")
    else:
        lst.append(nn.Linear(512,136))

    classifier = nn.Sequential(*lst)
    return classifier

class LandmarkNet(nn.Module):
    def __init__(self, resnet_dim = 101, classifier = None, pool_drop = True, requires_grad = False):
        super(LandmarkNet, self).__init__()
        self.extractor = get_feature_extractor(resnet_dim= resnet_dim, requires_grad=requires_grad, PoolDrop = pool_drop)

        if pool_drop:
            if classifier == None:
                self.classifier = get_classifier_deep()
            else:
                self.classifier = classifier
        else:#Legacy
            self.classifier = classifier
            print('Do not use it')

    def forward(self, x):
        if x.shape[1] ==1:#grayscale image
            x = x.repeat((1,3,1,1))
        return self.classifier(self.extractor(x))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        self.extractor = self.extractor.to(*args, **kwargs)
        return self

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

if __name__ == '__main__':
    import numpy as np
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms

    from label_io import read_data_names, read_labels, plot_image, chw, hwc
    from dataset import CoordDataset
    from label_transform import CoordCustomPad, CoordHorizontalFlip, CoordRandomRotate, \
        CoordLabelNormalize, CoordResize, CoordVerticalFlip

    #def label-to-image
    data_path = './train_images'
    label_path = './train_labels'
    labels = read_labels(label_location= label_path)
    data_names = read_data_names(label_location= label_path)

    batch_size = 8

    # transform = tr.Compose([
    #     tr.RandomRotation(30, expand = False)
    #     tr.ToTensor()
    # ])

    customTransforms = [
    CoordCustomPad(512 / 256),
    CoordResize((512, 256)),
    CoordLabelNormalize()
    ]

    # customTransforms = [
    #                     CoordRandomRotate(max_angle = 5, expand = True, is_random =False),
    #                     # CoordHorizontalFlip(0.5),
    #                     # CoordVerticalFlip(0.5),
    #                     CoordCustomPad(512 / 256),
    #                     CoordResize((512, 256)),
    #                     CoordLabelNormalize()
    #                     ]
    #
    # data_path = './resized_images'
    # label_path = './resized_labels'
    # for ind, data_name in enumerate(data_names):
    #     img = Image.open(os.path.join(data_path, data_names[ind]))
    #     seg = np.load(os.path.join(label_path, data_name + '.npy'))
    #     seg = tr.ToPILImage(seg)
    #     img, seg = customTransforms[0](img, seg)
    #     plt.figure()
    #     plt.subplot(211)
    #     plt.imshow(img)
    #     plt.subplot(212)
    #     plt.imshow(seg)
    #     plt.show()


    dset = CoordDataset(data_path, labels, data_names, transform_list=customTransforms)
    loader_original = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False)

    # for val_data in loader_original:
    #     imgs = val_data['image'].cpu().to(dtype=torch.float)
    #     labs = val_data['label'].cpu().to(dtype=torch.float)
    #     for ind in range(batch_size):
    #         img = imgs[ind]
    #         lab = labs[ind]

    _train_data = next(iter(loader_original))
    _imgs = _train_data['image']
    _labels = _train_data['label']
    _imgs = _imgs.repeat((1,3,1,1))
    ext = torchvision.models.resnet
    out = ext(_imgs)
    print(out.shape)
    #512 256은 8 2048 16 8 나옴



#class BoostLayer(nn.Module):
