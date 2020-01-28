import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from helpers import read_data_names, read_labels, plot_image, chw, hwc
from dataset import SpineDataset, CoordDataset
from model import SegmentNet, LandmarkNet, get_classifier_deep, SpinalStructured, get_classifier_conv
from postprocessing import CoordDataset, CoordCustomPad, CoordHorizontalFlip,CoordRandomRotate, CoordLabelNormalize, CoordResize, CoordVerticalFlip

'''
이제 끝 손대지말자

후처리구현

+ seg map이랑 해서 앙상블하자 이건 어쩔수없다.
일단 이것만해서 후처리하기 화요일 올인!

test loader랑 같이 해서 전체파일 수정하자!!

Segmap변환
일단 이때까지한거랑
랜덤 블러같은거
할수있는건 다 하자!!!!

BNC가 중요하다 Brightness and Contrast

segmain 따로만들기

Segment 아키텍쳐는 지금있는거로 ㄱㅊ 
아 귀찮아 그냥 이걸로 하자
세그멘트 짜지 말것!!
이거만 랩미팅에서 소개하고 이걸로 진행하기!!
세그멘트도 이런곳에서 똑같이 안됬다고 하자

'''





############################
#coord : labeled as  1 2
#                    3 4
#                    5 6
#then 1x 1y 2x 2y 3x 3y .....
#when it is reshaped into (-1,2): then coord[:,0] : (1x 2x 3x ...), [:,1] : (1y 2y 3y...)


####################################################
##      Parameters                                ##
####################################################

data_path = './highres_images'
label_path = './highres_labels'

val_ratio = 0.1
batch_size = 64


labels = read_labels(label_path)
data_names = read_data_names(label_path)

N_all = len(data_names)
N_val = int(N_all*val_ratio)
N_train = N_all - N_val

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
##############################################################################
##############################################################################


#############################################################################
#
#       Get DataLoader
#
#############################################################################
#get train and validation data set
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

RH = [
                       CoordRandomRotate(max_angle = 10, expand = False),
                        CoordHorizontalFlip(0.5),
                        CoordCustomPad(512 /256),
                        CoordResize((512,256)),
                        CoordLabelNormalize()
                       ]
NOPAD = [
                       CoordRandomRotate(max_angle = 10, expand = False),
                        CoordHorizontalFlip(0.5),
                        CoordResize((512,256)),
                        CoordLabelNormalize()
                       ]
UNF = [
                       CoordRandomRotate(max_angle = 10, expand = False),
                        CoordHorizontalFlip(0.5),
                        CoordCustomPad(512 /256, random = 'uniform'),
                        CoordResize((512,256)),
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

######################TRANSFORM 여기선 트랜스폼만
customTransforms = NOPAD
val_transforms = NOPAD_VAL
#########################
dset_train = CoordDataset(data_path, labels_train, data_names_train, transform_list=customTransforms)
dset_val = CoordDataset(data_path, labels_val, data_names_val, transform_list=val_transforms)

loader_train = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset=dset_val, batch_size=4, shuffle=False)

###################################
# TRAINING           ##############
###################################
from train import Trainer
state_dict = dict(num_epochs=2000, learning_rates=1e-5, save_every=50,
                  all_model_save=0.99,
                  is_lr_decay=True, lrdecay_thres=0.1, lrdecay_every=500,
                  model_save_path="./model", dropout_prob=0.5
                  )
state_dict['model_name'] = 'NOPAD_DEEP'
classifier = get_classifier_deep(dropout=state_dict['dropout_prob'])

model = LandmarkNet(PoolDrop=True, classifier=classifier).to(device)
trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=state_dict['learning_rates']),
#                  loader_train=loader_train, loader_val=loader_train, criterion=nn.SmoothL1Loss(), **state_dict)
                    loader_train = loader_train, loader_val = loader_val, criterion = nn.SmoothL1Loss(), ** state_dict)
# loader_train=loader_train, loader_val=loader_val, criterion=torch.nn.MSELoss(), **state_dict)
# tl = trainer.test(test_loader=loader_val, title=state_dict['model_name'] + '_init')
#trainer.test(test_loader=loader_val, load_model_name='RH_SM_all_ep500_tL4.09e-03_vL1.10e-03.model')
trainer.train()



