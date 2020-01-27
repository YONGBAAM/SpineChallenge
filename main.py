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
이제 끝 손대지말자

후처리구현

+ seg map이랑 해서 앙상블하자 이건 어쩔수없다.
일단 이것만해서 후처리하기 화요일 올인!

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

val_ratio = 0.0
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

RHV = [
                       CoordRandomRotate(max_angle = 10, expand = False),
                        CoordHorizontalFlip(0.5),
                        CoordVerticalFlip(0.5),
                        CoordCustomPad(512 /256),
                        CoordResize((512,256)),
                        CoordLabelNormalize()
                       ]

RH = [
                       CoordRandomRotate(max_angle = 10, expand = False),
                        CoordHorizontalFlip(0.5),
                        CoordCustomPad(512 /256),
                        CoordResize((512,256)),
                        CoordLabelNormalize()
                       ]

smooth = torch.nn.SmoothL1Loss()
MSE = torch.nn.MSELoss()

for is_lr_decay, name  in [(True,'FINALL_DEC'), (False, 'FINALL_NODEC')]:
    #고정
    customTransforms = RH
    crit = smooth

    dset_train = CoordDataset(data_path, labels_train, data_names_train, transform_list=customTransforms)
    dset_val = CoordDataset(data_path, labels_val, data_names_val, transform_list=[
        CoordCustomPad(512 / 256),
        CoordResize((512, 256)),
        CoordLabelNormalize()
    ])

    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset=dset_val, batch_size=4, shuffle=False)

    from postprocessing import check_dataset
    # check_dataset(loader_val)

    ###Testing the Training data
    # _train_data = next(iter(loader_train))
    # _imgs= _train_data['image']
    # _labels = _train_data['label']
    # _lb = _labels[0]
    # print(np.average(_lb))

    ###################################
    # TRAINING           ###############
    ###################################
    from train import Trainer

    is_lr_decay = is_lr_decay
    state_dict = dict(num_epochs=2000, learning_rates=1e-5, save_every=100,
                      all_model_save=0.99,
                      is_lr_decay=is_lr_decay, lrdecay_thres=0.1, lrdecay_every = 400,
                      model_save_path="D:\\TorchModels", dropout_prob=0.5
                      )

    state_dict['model_name'] = name
    #load_name = 'RH_SM_all_ep1999_tL1.45e-03_vL6.31e-04.model'

    model = LandmarkNet(PoolDrop=True, classifier=get_classifier_conv(dropout=state_dict['dropout_prob'])).to(device)
    trainer = Trainer(model=model,
                      optimizer=torch.optim.Adam(model.parameters(), lr=state_dict['learning_rates']),
                      loader_train=loader_train, loader_val=loader_train, criterion=crit, **state_dict)
                      #loader_train=loader_train, loader_val=loader_val, criterion=crit, **state_dict)
                      #loader_train=loader_train, loader_val=loader_val, criterion=torch.nn.MSELoss(), **state_dict)

    #trainer.load_model(title=load_name, all=True)
    #tl = trainer.test(test_loader=loader_val, title=state_dict['model_name'] + '_init')

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