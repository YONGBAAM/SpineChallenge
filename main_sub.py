import numpy as np
import os
import torch
import torch.nn as nn


from label_io import read_data_names, read_labels, plot_image, chw, hwc
from dataset import CoordDataset, get_loader_train, get_loader_test, get_loader_train_val
from model import SegmentNet, LandmarkNet, get_classifier_deep, SpinalStructured, get_classifier_conv
from train import Trainer

'''
git : revert to last commit 한다음 pull해주면 되겠다!

Labtop : 34 nopad
desktop : 101 nopad

cloud : 101 pad
34 nopad
101 nopad plus grad
each 2000

'''
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

###################################
# TRAINING           ##############
###################################

print('########################################################################')
print('THIS IS NOT MAIN FUNCTION')
print('########################################################################')

config = dict(num_epochs=2, learning_rates=1e-5, save_every=200,
              all_model_save=0.99,
              is_lr_decay=False, lrdecay_thres=0.1, lrdecay_every=200, lrdecay_window = 50,
              model_save_dest="./model", dropout_prob=0.5
              )

cl34 = nn.Sequential(*[#512 16 8 for 34
    nn.Conv2d(512,128,1),
    nn.BatchNorm2d(128),
    nn.ReLU(),

    nn.Conv2d(128,128,3,padding = 1),
    nn.BatchNorm2d(128),
    nn.ReLU(),

    nn.Conv2d(128,128,kernel_size=4, stride=2, padding =1),
    nn.BatchNorm2d(128),
    nn.ReLU(),

    nn.Flatten(),
    nn.Linear(128*8*4,4096),
    nn.ReLU(),
    nn.Dropout(config['dropout_prob']),

    nn.Linear(4096,136)

])

############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.0
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'pool'

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad_val', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm = 'nopad_val', batch_size = 1, shuffle = False )
####    MODEL 101_deep
#model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow

cl_for_flat = nn.Sequential(*[
    nn.Flatten(),   #2048

    nn.Linear(2048,4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096,4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096,1024),
    nn.BatchNorm2d(1024),
    nn.ReLU(),

    nn.Linear(1024,136)
])
model = LandmarkNet(resnet_dim=101, classifier = cl_for_flat, requires_grad=False, pool_drop=False).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train = loader_train, loader_val = loader_val, criterion = nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.0
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'pool_RH'

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow

cl_for_flat = nn.Sequential(*[
    nn.Flatten(),  # 2048

    nn.Linear(2048, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096, 1024),
    nn.BatchNorm2d(1024),
    nn.ReLU(),

    nn.Linear(1024, 136)
])
model = LandmarkNet(resnet_dim=101, classifier=cl_for_flat, requires_grad=False, pool_drop=False).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
config['dropout_prob'] = 0.0
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'nopool_RH'

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow

cl_for_flat = nn.Sequential(*[
    nn.Flatten(),  # 2048

    nn.Linear(2048, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096, 1024),
    nn.BatchNorm2d(1024),
    nn.ReLU(),

    nn.Linear(1024, 136)
])
model = LandmarkNet(resnet_dim=101, classifier=cl_for_flat, requires_grad=False, pool_drop=False).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
config['dropout_prob'] = 0.0
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'pooldrop'

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow

cl_for_flat = nn.Sequential(*[
    nn.Flatten(),  # 2048

    nn.Linear(2048, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(),

    nn.Linear(4096, 1024),
    nn.BatchNorm2d(1024),
    nn.ReLU(),

    nn.Linear(1024, 136)
])
model = LandmarkNet(resnet_dim=101, classifier=get_classifier_conv(dropout=config['dropout_prob']),
                    requires_grad=False, pool_drop=True).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()



############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.0
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'nodropout'

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow
model = LandmarkNet(resnet_dim=101, classifier=get_classifier_deep(dropout=config['dropout_prob']),
                    requires_grad=False, pool_drop=True).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.5
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'nodeep'

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow
model = LandmarkNet(resnet_dim=101, classifier=get_classifier_conv(dropout=config['dropout_prob']),
                    requires_grad=False, pool_drop=True).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.5
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'pad'

####    DataLoader
loader_train = get_loader_train(tfm = 'rhp', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='pad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow
model = LandmarkNet(resnet_dim=101, classifier=get_classifier_deep(dropout=config['dropout_prob']),
                    requires_grad=False, pool_drop=True).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.5
config['is_lr_decay'] = True
batch_size = 96
config['model_name'] = 'with_decay'
config['num_epochs'] = 2000

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow
model = LandmarkNet(resnet_dim=101, classifier=get_classifier_deep(dropout=config['dropout_prob']),
                    requires_grad=False, pool_drop=True).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()

############################################################################
############################################################################
############################################################################
config['dropout_prob'] = 0.5
config['is_lr_decay'] = False
batch_size = 96
config['model_name'] = 'sub_final'
config['num_epochs'] = 2000

####    DataLoader
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm='nopad_val', batch_size=1, shuffle=False)
####    MODEL 101_deep
# model = LandmarkNet(resnet_dim=101, classifier = get_classifier_deep(config['dropout_prob']), requires_grad=True).to(device)
####    MODEL 34_swallow
model = LandmarkNet(resnet_dim=101, classifier=get_classifier_deep(dropout=config['dropout_prob']),
                    requires_grad=False, pool_drop=True).to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train=loader_train, loader_val=loader_val, criterion=nn.MSELoss(), **config)
trainer.train()
trainer.test()
