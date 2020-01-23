import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from helpers import read_data_names, read_labels, plot_image, chw, hwc
from dataset import SpineDataset
from model import SegmentNet, LandmarkNet, get_classifier, SpinalStructured


############################
#coord : labeled as  1 2
#                    3 4
#                    5 6
#then 1x 1y 2x 2y 3x 3y .....
#when it is reshaped into (-1,2): then coord[:,0] : (1x 2x 3x ...), [:,1] : (1y 2y 3y...)
#
#################################


###################################
#       System & Data Parameters  #
###################################
RESET_PERM = False
data_path = './resized_images'
label_path = './resized_labels'
save_path = './model'

is_segment = False ######for linear model

val_ratio = 0.05
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

##Get DataLoader
#get label and dataname list
coords_rel = read_labels(label_path, relative = True)
data_names = read_data_names(label_path)

N_all = len(data_names)
N_val = int(N_all*val_ratio)
N_train = N_all - N_val

#get train and validation data set
data_names_train = []
data_names_val = []
coords_train = []
coords_val = []

if not os.path.exists(os.path.join('./', 'val_permutation.npy')) or RESET_PERM:
    permutation = np.random.permutation(N_all)
    np.save(os.path.join('./', 'val_permutation.npy'), permutation)
else:
    permutation = np.load(os.path.join('./', 'val_permutation.npy'))

for ind in permutation[:N_train]:
    data_names_train.append(data_names[ind])
    coords_train.append(coords_rel[ind])
coords_train = np.asarray(coords_train)

for ind in permutation[N_train:]:
    data_names_val.append(data_names[ind])
    coords_val.append(coords_rel[ind])
coords_val = np.asarray(coords_val)

dset_train = SpineDataset(data_path, label_path, data_names_train, coords_rel=coords_train, transform=transform
                          , is_segment = is_segment)
loader_train = DataLoader(dataset = dset_train, batch_size =batch_size, shuffle = True)

dset_val = SpineDataset(data_path, label_path, data_names_val,coords_rel=coords_val, transform=transform
                        , is_segment = is_segment)
loader_val = DataLoader(dataset = dset_val, batch_size = batch_size, shuffle = False)

#######Displaying the training data

_train_data = next(iter(loader_train))
_imgs= _train_data['image']
_labels = _train_data['label']

# if batch_size >4:
#     for i in range(4):
#         plt.subplot(221 + i)
#         label = np.asarray(_labels[i])
#         image = np.asarray(_imgs[i])
#         plot_image(image, segmap=label)
#     plt.show()

###################################
#TRAINING           ###############
###################################
from train import Trainer

#Training parameter define
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

state_dict = {'testmode' : False, 'no_train' : N_train, 'no_val' : N_val,
              'num_epochs': 200, 'learning_rates' : 3e-5,
              'save_every' : 25, 'all_model_save' : 0.95,
              'is_lr_decay' : False
              }

####################################################
##FOR OVERFIT                                      #
####################################################
#loader_val = loader_train
####################################################
classifiers = []
h1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 136+8),
    SpinalStructured()
)
h2 = get_classifier()
h3 = nn.Sequential(
        nn.Flatten(),#2048 for resnet 101
        nn.Linear(2048,4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 136 + 8),
        SpinalStructured(output_dim=68)
    )
h4 = nn.Sequential(
        nn.Flatten(),#2048 for resnet 101
        nn.Linear(2048,4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512, 136 + 8),
        SpinalStructured(output_dim=68)
    )
for ind, c in enumerate([h1,h4]):
    model = LandmarkNet(classifier = c)
    model = model.to(device)
    state_dict['model_name'] = 'cl{}'.format(4-ind)
    optim = torch.optim.Adam(model.parameters(), lr=state_dict['learning_rates'])
    crit = torch.nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optim, loader_train=loader_train,
                      loader_val=loader_val, criterion=crit, **state_dict)

    trainer.train()


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
