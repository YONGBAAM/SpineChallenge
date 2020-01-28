import pandas as pd
import numpy as np
import os
import scipy.io as spio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from helpers import plot_image
import torch
import torch.nn as nn
from dataset import SpineDataset
import torchvision.transforms as transforms
from model import get_feature_extractor
import datetime
from helpers import write_labels

from collections.abc import Iterable

#Helpers
def get_time_char():
    tm = datetime.datetime.now()
    return '{}/{}_{}:{}'.format(tm.month, tm.day, tm.hour, tm.minute)

class Trainer():
    def __init__(self, model, optimizer,loader_train, loader_val, criterion
                 , **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.criterion = criterion

        #DEFALUT value means test condition
        self.num_epochs = kwargs.get('num_epochs', None)
        self.testmode = kwargs.get('testmode', False)
        self.no_train = loader_train.dataset.size
        self.no_val = loader_val.dataset.size

        self.learning_rates = kwargs.get('learning_rates', 1e-4)
        self.lrdecay_thres = kwargs.get('lrdecay_thres', 0.1)
        self.is_lr_decay = kwargs.get('is_lr_decay', False)
        self.lrdecay_every = kwargs.get('lrdecay_every', 500)
        self.last_lrdecay = 0
        self.dropout_prob = kwargs.get('dropout_prob', 0)

        self.save_every = kwargs.get('save_every', 1)
        self.all_model_save = kwargs.get('all_model_save', 0)

        #self.reg_factor = kwargs.get('reg_factor', 0)
        self.reg_crit = kwargs.get('reg_crit', nn.MSELoss())

        if type(model).__name__ == 'LandmarkNet':
            self.is_landmark =True
        elif type(model).__name__ == 'SegmentNet':
            self.is_landmark = False
        else:
            print('model is not detecetd, is_landmark is None')
            self.is_landmark = None

        self.model_name = kwargs.get('model_name', None)
        if self.model_name == None:
            tm = datetime.datetime.now()
            if self.is_landmark == True:
                prefix = 'ldm'
            elif self.is_landmark == False:
                prefix = 'seg'
            else:
                prefix = 'NN'
            self.model_name = prefix + '_' + '{}{}{}'.format(tm.day, tm.hour, tm.minute)

        #self.data_path_train = kwargs.pop(['data_path_train'])
        #self.label_path_train = kwargs.pop(['label_path_train'])
        save_loc = kwargs.get('model_save_path', './model')
        self.model_save_path = os.path.join(save_loc, self.model_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.device = next(self.model.parameters()).device

        self.loss_tracks = []
        self.loss_list = []
        self.val_loss_list = []

        self.log = []#50번째 save 했고 등등....

        self.init_log(**kwargs)

    def _train_epoch(self):
        model = self.model
        model.train()
        losses = []

        for train_data in self.loader_train:
            #print('train_ep')
            model.zero_grad()
            imgs = train_data['image'].to(self.device, dtype=torch.float)
            labels = train_data['label'].to(self.device, dtype=torch.float)
            out = model(imgs)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if self.testmode == True:
                print('loss%.2e'%(loss.item()))
                break

        return losses

    def train(self):
        print("Model Name : {}, Training Started".format(self.model_name))
        print("VRAM : {}GB".format(torch.cuda.get_device_properties(self.device).total_memory/1000000))


        save_every = self.save_every
        all_model_save_thr = self.all_model_save
        num_epochs = self.num_epochs
        save_morethan = num_epochs * all_model_save_thr

        for ep in range(num_epochs):
            losses = self._train_epoch()
            self.loss_list.append(np.average(losses))
            self.loss_tracks.append(losses)


            val_losses = self.validate()
            self.val_loss_list.append(np.average(val_losses))
            print("ep %d, loss_t %.2e, loss_v %.2e"%(ep, np.average(losses), np.average(val_losses)))
            if self.is_lr_decay:
                self.lr_decay()

            if ep > save_morethan or ep%save_every == 0 or ep == num_epochs -1:
                title = self.model_name + '_ep%d' % (ep)

                val_losses = self.validate(title_if_plot_save = title)
                self.save_model(title + '_tL%.2e_vL%.2e'%(np.average(losses), np.average(val_losses)), all = True)
                self.update_log(logline = 'ep{} model saved : {}'.format(ep, title))
                self.save_loss(title=title)
                self.save_log(title=title)

    def save_model(self, title, all = False):
        if  not title[-6:] == '.model':
            title = title + '.model'

        if all:
            # for line in list(self.model.state_dict()):
            #     print(line)
            #     self.update_log(line)
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, title))
        else:
            print('Deprecated, save all state dict')
            # if self.is_landmark:
            #     torch.save(self.model.classifier.state_dict(), os.path.join(self.model_save_path, title))
            # else:
            #     torch.save(self.model.state_dict(), os.path.join(self.model_save_path, title))




    def load_model(self, title, all = False):
        if  not title[-6:] == '.model':
            title = title + '.model'
        #현재 path에 모델 있으면 불러오기
        #아니면 올라가서 찾자

        if not all:
            print('Deprecated, load all state dict')
            return

        if os.path.exists(os.path.join(self.model_save_path, title)):
            model_load_path = os.path.join(self.model_save_path, title)
            print('loading {}'.format(model_load_path))
            self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, title)))
        else:
            up_path = self.model_save_path.split('\\')
            up_path = '/'.join(up_path[:-1])
            folder = title.split('ep')[0]
            folder = folder[:-1]
            model_load_path = os.path.join(up_path,folder, title)
            print('loading {}'.format(model_load_path))
            self.model.load_state_dict(torch.load(model_load_path))

    def save_config(self, title):
        #save the parameters of training
        #lr, adam/sgd...
        pass

    def save_loss(self, title):
        arr = np.asarray([self.loss_list, self.val_loss_list])
        arr = arr.T

        df = pd.DataFrame(arr, columns = {'t_loss', 'v_loss'})
        df.insert(loc = 2, column = 'track', value= self.loss_tracks)

        if not title[-4:] == '.csv':
            title = title + '.csv'

        df.to_csv(os.path.join(self.model_save_path, title ))

    def validate(self, title_if_plot_save = None):
        #클래스 함수에는 이런 지저분한거 없게 하자!!!!
        self.model.eval()
        val_losses = []
        val_labels = []
        img_list = []
        true_labels = []
        for val_data in self.loader_val:
            self.model.zero_grad()
            imgs = val_data['image'].to(self.device, dtype=torch.float)
            labels = val_data['label'].to(self.device, dtype=torch.float)
            out = self.model(imgs)
            loss = self.criterion(out, labels)
            val_losses.append(loss.item())

            #for save validation
            val_labels.append(np.asarray(out.cpu().detach()))
            img_list.append(np.asarray(imgs.cpu().detach()))
            true_labels.append(np.asarray(labels.cpu().detach()))

        val_labels = np.concatenate(val_labels, axis = 0)
        imgs = np.concatenate(img_list, axis = 0)
        true_labels = np.concatenate(true_labels, axis = 0)

        if title_if_plot_save is not None:
            # validate data 검증
            perm = np.random.permutation(self.no_val)
            plt.figure()
            for i in range(4):
                ind = perm[i]

                plt.subplot(221 + i)
                if self.is_landmark == True:
                    #plot_image(imgs[ind], ref_coord=true_labels[ind], coord = val_labels[ind])
                    plot_image(imgs[ind], coord=true_labels[ind], ref_coord=val_labels[ind])
                elif self.is_landmark == False:
                    plot_image(imgs[ind], segmap = true_labels[ind], ref_segmap=val_labels[ind])
                plt.title('val {}'.format(perm[i]))

            plt.savefig(os.path.join(self.model_save_path, title_if_plot_save + '.png'))
            plt.close()

        return val_losses

    def test(self, test_path = None, test_loader = None, load_model_name = None, title = None):

        ##Getting save path and model load
        if load_model_name is not None:
            if os.path.exists(os.path.join(self.model_save_path, load_model_name)):
                #current path, load model
                model_load_path = os.path.join(self.model_save_path, load_model_name)
                print('loading {}'.format(model_load_path))
                result_save_path = self.model_save_path
                self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, load_model_name)))
            else:#new path, load model
                up_path = self.model_save_path.split('\\')
                up_path = '/'.join(up_path[:-1])
                folder = load_model_name.split('ep')[0]
                folder = folder[:-1]
                ep_no = load_model_name.split('ep')[1].split('_')[0]
                model_load_path = os.path.join(up_path, folder, load_model_name)
                print('loading {}'.format(model_load_path))
                result_save_path = up_path + '/' + folder + '_ep' + ep_no
                self.model.load_state_dict(torch.load(model_load_path))
        else:
            #no load, existing model
            result_save_path = self.model_save_path
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        print('save result in {}'.format(result_save_path))


        if test_loader == None:
            #make testloader
            #val로 하고싶으면 loader 주면 된다!!!!!
            print('not implemented yet :D')



        test_crit = nn.MSELoss()

        self.model.eval()
        test_losses = []
        test_labels = []
        img_list = []
        true_labels = []
        for test_data in test_loader:
            self.model.zero_grad()
            imgs = test_data['image'].to(self.device, dtype=torch.float)
            labels = test_data['label'].to(self.device, dtype=torch.float)
            out = self.model(imgs)
            loss = test_crit(out, labels)
            test_losses.append(loss.item())

            _,_,H,W = imgs.shape

            # for save validation
            test_labels.append(np.asarray(out.cpu().detach()))
            img_list.append(np.asarray(imgs.cpu().detach()))
            true_labels.append(np.asarray(labels.cpu().detach()))

        test_labels = np.concatenate(test_labels, axis=0)
        imgs = np.concatenate(img_list, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        if title is None:
            title = folder + '_ep' + ep_no + '_t'

        #save image result for test
        for ind in range(imgs.shape[0]):
            plt.figure()
            if self.is_landmark == True:
                plot_image(imgs[ind], coord=true_labels[ind], ref_coord=test_labels[ind])
            elif self.is_landmark == False:
                plot_image(imgs[ind], segmap=test_labels[ind], ref_segmap=true_labels[ind])
            plt.title(title + '_{}'.format(ind))
            plt.savefig(os.path.join(result_save_path, title + '_{}.jpg'.format(ind)))
            plt.close()

        #save absolute label result
        N = test_labels.shape[0]
        labels_abs = test_labels.reshape(N,-1,2)
        labels_abs[:,:,0] *= W
        labels_abs[:,:,1] *= H
        labels_abs = labels_abs.reshape(N,-1)
        write_labels(labels_abs, result_save_path)

        print('test MSE loss %.2e'%(np.average(test_losses)))
        return test_losses

        #log save
    def init_log(self, **kwargs):
        self.log.append((get_time_char(), 'initialize program'))
        for k, v in kwargs.items():
            self.log.append((get_time_char(), '{}:{}'.format(k,v)))

    def update_log(self, logline):
        self.log.append((get_time_char(), logline))

    def save_log(self, title = None):
        if title is None:
            title = 'log'

        with open(os.path.join(self.model_save_path, title + '.txt') ,'w') as f:
            for line in self.log:
                f.write(line[0] +'\t' +  line[1] + '\n')


    def lr_decay(self):
        current_ep = len(self.loss_list)
        if current_ep > self.lrdecay_every + self.last_lrdecay or self.testmode:
            window = int(self.save_every/2)
            before_average = np.average(self.loss_list[current_ep -1 - self.save_every:
                                        current_ep -1 - self.save_every + window])
            recent_loss = np.array(self.loss_list[current_ep-window:current_ep])
            diff = np.abs(recent_loss - before_average)/before_average
            more_change = diff > self.lrdecay_thres

            if sum(more_change) ==0 or self.testmode:#no change
                self.learning_rates /= 3
                if type(self.optimizer) == type(torch.optim.Adam(self.model.parameters(), lr = 0.001)):
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rates)
                elif type(self.optimizer) == type(torch.optim.RMSprop(self.model.parameters(), lr = self.learning_rates)):
                    self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = self.learning_rates)
                # for param_group in self.optimizer.param_groups:
                #     param_group['lr'] /= 1.58 #pow(10,0.333)

                self.last_lrdecay = current_ep
                print('lr decayed')
                self.update_log('lr decayed to %.2e'%(self.learning_rates))
                return True
            else:return False
        else:return False


'''     
class MetricTracker(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.items = []

    def get_items(self, batch_per_ep = None):
        size = len(self.items)
        x = np.arange(1,size+1)

        if batch_per_ep is not None:
            x /= batch_per_ep

        xy = [np.asarray(list(x)), np.asarray(self.items)]
        return np.asarray(xy)

    def get_smoothed_list(self, window = 5, batch_per_ep = None):
        size = len(self.items)
        if window >size:
            return None
        else:
            item_s = np.zeros(size)
            for i in range(window -1, item_s.shape[0]):
                item_s[i] = np.sum(self.item[i-(window -1):i])
            x = np.arange(1, size + 1)

            if batch_per_ep is not None:
                x /= batch_per_ep

            xy = [np.asarray(list(x)), np.asarray(item_s)]
        return xy

    def update(self, value_list):
        if not type(value_list) == type([]):
            value_list = [value_list]
        else:
            self.items.extend(value_list)
        return self

    def avg(self):
        return np.average(self.items)







'''