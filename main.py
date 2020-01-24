import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from helpers import read_data_names, read_labels, plot_image, chw, hwc
from dataset import SpineDataset, CoordDataset
from model import SegmentNet, LandmarkNet, get_classifier, SpinalStructured, get_classifier_conv
from postprocessing import CoordDataset, CoordCustomPad, CoordHorizontalFlip,CoordRandomRotate, CoordLabelNormalize, CoordResize, CoordVerticalFlip

'''
프로젝트관련 진행사항은 여기다가 적자
당부 : 메인 두개만들지 말기!! 아래 주석으로
좌표추정은 버리자. 그냥 한계 명시하기.
지금 돌려놓은거(spinestructure, dropout)만 쓰기

나머지 연휴때는 그냥 seg만 주구장창 돌리자.
Seg는 좌표 벗어난거 확인 안해도 됨!!걍해

landmark를 추정하는게 맞을까?
ㅇㅇ맞아 더 하지마

일단 랜드마크가지고 하고 후처리를 하던가 하기

후처리할거면 걍 세그멘테이션으로 하는게 낫지않나?

일단 오늘 4개 돌려놓은거 결과확인하고 로테이션 손보고
그다음 마지막거 4000 + dropout + aug 해서 트레이닝!!
이거랑 세그멘테이션도 완전트레이닝 둘다!!


로테노 로테손본거 세그 3개 트레이닝 완전
세그는 로테 손안봐도됨 걍 잘라
돌려놓고 손 떼!!!!!!!!!!!!!!!!!

모델 토치모델 디드라이브로 변경




\
\
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



RESET_PERM = False
data_path = './highres_images'
label_path = './highres_labels'
save_path = './model'

val_ratio = 0.05
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

if not os.path.exists(os.path.join('./', 'val_permutation.npy')) or RESET_PERM:
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

customTransforms = [
#                       CoordRandomRotate(max_angle = 5, expand = True),
                        CoordHorizontalFlip(0.5),
#                        CoordVerticalFlip(0.5),
                        CoordCustomPad(512 /256),
                        CoordResize((512,256)),
                        CoordLabelNormalize()
                       ]

dset_train = CoordDataset(data_path, labels_train, data_names_train, transform_list=customTransforms)
dset_val = CoordDataset(data_path, labels_val, data_names_val, transform_list= [
    CoordCustomPad(512 / 256),
    CoordResize((512, 256)),
    CoordLabelNormalize()
])

loader_train = DataLoader(dataset = dset_train, batch_size =batch_size, shuffle = True)
loader_val = DataLoader(dataset = dset_val, batch_size = 4, shuffle = False)

from postprocessing import check_dataset
#check_dataset(loader_val)

###Testing the Training data
# _train_data = next(iter(loader_train))
# _imgs= _train_data['image']
# _labels = _train_data['label']
# _lb = _labels[0]
# print(np.average(_lb))

###################################
#TRAINING           ###############
###################################
from train import Trainer

state_dict = {'testmode' : False, 'no_train' : N_train, 'no_val' : N_val,
              'num_epochs': 1500, 'learning_rates' : 1e-5,
              'save_every' : 50, 'all_model_save' : 0.95,
              'is_lr_decay' : True, 'lrdecay_thres' : 0.1
              }


# for prob in [0.2,0.5]:
#     classifier = get_classifier_conv(dropout=prob)
#     model = LandmarkNet(PoolDrop=True, classifier=classifier)
#     model = model.to(device)
#     state_dict['model_name'] = 'conv_p{}'.format(prob)
#     optim = torch.optim.RMSprop(model.parameters(), lr=state_dict['learning_rates'])
#     crit = torch.nn.MSELoss()
#     trainer = Trainer(model=model, optimizer=optim, loader_train=loader_train,
#                       loader_val=loader_val, criterion=crit, **state_dict)
#     trainer.train()


name = 'conv_p0.5_from350_ep400_tL3.02e-03_vL1.64e-03.model'

for prob, sp in [(0.5,False)]:
    classifier = get_classifier_conv(dropout=prob)
    model = LandmarkNet(PoolDrop=True, classifier=classifier)
    model = model.to(device)
    state_dict['model_name'] = 'conv_p{}_700'.format(prob)
    optim = torch.optim.Adam(model.parameters(), lr=state_dict['learning_rates'])
    crit = torch.nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optim, loader_train=loader_train,
                      loader_val=loader_val, criterion=crit, **state_dict)
    trainer.load_model(name)

    trainer.train()


'''
#########################################33
#
#       Upper : Linear
#       For double training
#       Lower : Segment
#
#############################################RESET_PERM = False
is_segment = True
batch_size = 8
coords_rel = read_labels(label_path)

#나중에 달라질수도 있으므로
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dset_train.is_segment = is_segment
loader_train = DataLoader(dataset = dset_train, batch_size =batch_size, shuffle = True)

dset_val.is_segment = is_segment
loader_val = DataLoader(dataset = dset_val, batch_size = batch_size, shuffle = False)

###################################
#TRAINING           ###############
###################################
state_dict = {'testmode' : False, 'no_train' : N_train, 'no_val' : N_val,
              'num_epochs': 2000, 'learning_rates' : 0.0003,
              'save_every' : 50, 'all_model_save' : 0.95,
              'is_lr_decay' : True
              }

# for lr in [3e-4]:
#     model = SegmentNet().to(device)
#     state_dict['model_name'] = '211745_from2300'
#     state_dict['learning_rates'] = lr
#     optim = torch.optim.Adam(model.parameters(), lr=state_dict['learning_rates'])
#     crit = torch.nn.MSELoss()
#     trainer = Trainer(model=model, optimizer=optim, loader_train=loader_train,
#                       loader_val=loader_val, criterion=crit, **state_dict)
#
#     trainer.load_model('torch_211745_from2000_ep300_tL1.60e-03_vL2.46e-02.model')
#     trainer.train()
'''