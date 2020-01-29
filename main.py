import numpy as np
import os
import torch
import torch.nn as nn


from label_io import read_data_names, read_labels, plot_image, chw, hwc
from dataset import CoordDataset, get_loader_train, get_loader_test
from model import SegmentNet, LandmarkNet, get_classifier_deep, SpinalStructured, get_classifier_conv


'''

세이브시 trainer 그대로 세이브
load시 그대로 할수있게
abort랑 이어서트레이닝 구현 
그냥optim이랑 모델만 세이브하자.

git : revert to last commit 한다음 pull해주면 되겠다!

abort 시그널

classifier dim을 키우고 dropout을 키워버릴까??


'''

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
batch_size = 64
loader_train = get_loader_train(tfm = 'nopad', batch_size=batch_size, shuffle=True)
loader_val = get_loader_test(tfm = 'nopad_val', batch_size = 1, shuffle = False )

###################################
# TRAINING           ##############
###################################
from train import Trainer
state_dict = dict(num_epochs=20, learning_rates=1e-5, save_every=20,
                  all_model_save=0.99,
                  is_lr_decay=True, lrdecay_thres=20, lrdecay_every=20,
                  model_save_dest="./model", dropout_prob=0.5
                  )
state_dict['model_name'] = 'NEW_TEST'
classifier = get_classifier_deep(dropout=state_dict['dropout_prob'])

model = LandmarkNet(PoolDrop=True, classifier=classifier).to(device)
trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=state_dict['learning_rates']),
                    loader_train = loader_train, loader_val = loader_val, criterion = nn.SmoothL1Loss(), ** state_dict)

#trainer.test(test_loader=loader_val, load_model_name='NEW_TEST_ep4_tL1.65e+16_vL1.55e+00.tar')
trainer.load_model('NEW_TEST_ep3_tL6.69e+12_vL1.36e+00.tar', model_only=False)
trainer.train()



