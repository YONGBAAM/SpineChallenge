import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from label_io import plot_image
import torch
import torch.nn as nn
import datetime
from label_io import write_labels

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
        self.start_ep = kwargs.get('start_ep', 0)
        self.current_ep = self.start_ep
        self.num_epochs = kwargs.get('num_epochs', None)

        self.testmode = kwargs.get('testmode', False)

        self.no_train = loader_train.dataset.size
        self.no_val = loader_val.dataset.size

        self.learning_rates = kwargs.get('learning_rates', 1e-5)
        self.lrdecay_thres = kwargs.get('lrdecay_thres', 0.1)
        self.is_lr_decay = kwargs.get('is_lr_decay', False)
        self.lrdecay_every = kwargs.get('lrdecay_every', 500)
        self.lrdecay_window = kwargs.get('lrdecay_window', 50)
        self.last_lrdecay = -200

        self.dropout_prob = kwargs.get('dropout_prob', 0)

        self.save_every = kwargs.get('save_every', 1)
        self.all_model_save = kwargs.get('all_model_save', 0)

        #self.reg_factor = kwargs.get('reg_factor', 0)
        self.reg_crit = kwargs.get('reg_crit', nn.MSELoss())

        self.model_name = kwargs.get('model_name', None)
        if self.model_name == None:
            tm = datetime.datetime.now()
            self.model_name = '{}{}{}'.format(tm.day, tm.hour, tm.minute)

        save_loc = kwargs.get('model_save_dest', './model')
        self.model_save_dest = os.path.join(save_loc, self.model_name)
        if not os.path.exists(self.model_save_dest):
            os.makedirs(self.model_save_dest)

        self.device = next(self.model.parameters()).device

        self.loss_tracks = []
        self.loss_list = []
        self.val_loss_list = []

        self.log = []#50번째 save 했고 등등....

        self.init_log(**kwargs)

        with open('./signal.txt', 'w') as f:
            f.write('#abort')

    #testing the
    def _train_epoch(self):
        model = self.model
        model.train()
        losses = []

        for train_data in self.loader_train:
            model.zero_grad()
            imgs = train_data['image'].to(self.device, dtype=torch.float)
            labels = train_data['label'].to(self.device, dtype=torch.float)
            out = model(imgs)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def train(self):
        print("Model Name : {}, Training Started".format(self.model_name))
        print("VRAM : {}GB".format(torch.cuda.get_device_properties(self.device).total_memory/1000000))

        save_every = self.save_every
        all_model_save_thr = self.all_model_save
        num_epochs = self.num_epochs
        save_morethan = num_epochs * all_model_save_thr
        abort = False

        for self.current_ep in range(self.start_ep, num_epochs):

            losses = self._train_epoch()
            self.loss_list.append(np.average(losses))
            self.loss_tracks.append(losses)

            val_losses = self.validate()
            self.val_loss_list.append(np.average(val_losses))
            show_string = ("{} %d: l_t %.2e, l_v %.2e"%(self.current_ep, np.average(losses), np.average(val_losses))).format(self.model_name)
            print(show_string)
            if self.is_lr_decay:
                self.lr_decay()

            # get abort signal
            with open('./signal.txt') as f:
                a = f.readline()
            if a == 'abort':
                abort = True
                print("ABORT signal detected")

            if self.current_ep > save_morethan or self.current_ep%save_every == 0 \
                    or self.current_ep == num_epochs -1 or abort:
                title = self.model_name + '_ep%d' % (self.current_ep)

                val_losses = self.validate(title_if_plot_save = title)
                #val_losses = [0]
                self.save_model(title + '_tL%.2e_vL%.2e'%(np.average(losses), np.average(val_losses)), model_only=False)
                self.update_log(logline = 'ep{} model saved\tTL : {}'.format(self.current_ep, title))
                self.update_log(logline = "ep %d, loss_t %.2e, loss_v %.2e"%(self.current_ep, np.average(losses), np.average(val_losses)))
                self.save_loss(title=title)
                self.save_log(title=title)

            if abort: break


    #Refactoring complete
    def save_model(self, title, model_only = False):
        if model_only:
            print('Saving model only, USE model_only = False afterwords')
            if not title[-6:] == '.model':
                title = title + '.model'
            torch.save(self.model.state_dict(), os.path.join(self.model_save_dest, title))
        else:
            save_dict = dict(current_ep = self.current_ep, model = self.model.state_dict(),
                             optimizer = self.optimizer.state_dict())
            if not title[-4:] == '.tar':
                title = title + '.tar'
            torch.save(save_dict, os.path.join(self.model_save_dest, title))

    def _load_model(self, path, model_only):
        if model_only:
            self.model.load_state_dict(torch.load(path))
        else:
            loaddata = torch.load(path)
            self.start_ep = loaddata['current_ep'] + 1
            self.last_lrdecay = self.start_ep
            self.model.load_state_dict(loaddata['model'])
            self.optimizer.load_state_dict(loaddata['optimizer'])
            for param_group in self.optimizer.param_groups:
                self.learning_rates = param_group['lr']

    #latest model load?
    def load_model(self, title, model_only = False):
        if model_only == False:
            if not title[-4:] == '.tar':
                title = title + '.tar'

        #get model_load_path
        #절대경로 모드
        if os.path.exists(title):
            model_load_path = title
        #현재 path에 모델 있으면 불러오기
        elif os.path.exists(os.path.join(self.model_save_dest, title)):
            model_load_path = os.path.join(self.model_save_dest, title)
        else: #아니면 올라가서 찾자
            up_path = self.model_save_dest.split('\\')
            up_path = '/'.join(up_path[:-1])
            folder = title.split('ep')[0]
            folder = folder[:-1]
            model_load_path = os.path.join(up_path,folder, title)

        print('Loading {}'.format(model_load_path))
        self._load_model(path=model_load_path, model_only=model_only)
        if model_only:
            print('Loading model only, USE model_only = False afterwords')

    def save_loss(self, title):
        arr = np.asarray([self.loss_list, self.val_loss_list])
        arr = arr.T

        df = pd.DataFrame(arr, columns = {'t_loss', 'v_loss'})
        df.insert(loc = 2, column = 'track', value= self.loss_tracks)

        if not title[-4:] == '.csv':
            title = title + '.csv'

        df.to_csv(os.path.join(self.model_save_dest, title))

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
            for i in range(8):
                ind = perm[i]
                plt.subplot(241 + i)
                plot_image(imgs[ind], coord_red=true_labels[ind], coord_gr=val_labels[ind])
                plt.title('val {}'.format(perm[i]))
            plt.savefig(os.path.join(self.model_save_dest, title_if_plot_save + '.png'))
            plt.close()

        return val_losses

    def test(self, test_loader = None, load_model_name = None, title = None, save_image = False):

        ##Getting save path and model load
        if load_model_name is not None:
            up_path = self.model_save_dest.split('\\')
            up_path = '/'.join(up_path[:-1])
            folder = load_model_name.split('ep')[0]
            folder = folder[:-1]
            ep_no = load_model_name.split('ep')[1].split('_')[0]
            result_save_path = up_path + '/' + folder + '_ep' + ep_no
            self.load_model(load_model_name)
        else:
            #no load, existing model
            result_save_path = self.model_save_dest

        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        print('Save Relative labels in {}'.format(result_save_path))

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
        if save_image:
            for ind in range(imgs.shape[0]):
                plt.figure()
                plot_image(imgs[ind], coord_red=true_labels[ind], coord_gr=test_labels[ind])
                plt.title(title + '_{}'.format(ind))
                plt.savefig(os.path.join(result_save_path, title + '_{}.png'.format(ind)))
                plt.close()

        #save absolute label result
        test_labels = test_labels.reshape(-1,136)
        write_labels(test_labels, result_save_path, title = 'labels_pred_rel')

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

        with open(os.path.join(self.model_save_dest, title + '.txt') , 'w') as f:
            for line in self.log:
                f.write(line[0] +'\t' +  line[1] + '\n')

    def _lrdeday(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        lr = lr/1.58
        self.learning_rates = lr
        if type(self.optimizer) == type(torch.optim.Adam(self.model.parameters(), lr=0.001)):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rates)
        elif type(self.optimizer) == type(torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rates)):
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rates)
        self.last_lrdecay = self.current_ep
        self.update_log('lr decayed to %.2e' % (self.learning_rates))
        print('lr decayed to %.2e' % (self.learning_rates))


    def lr_decay(self):
        ##_lrdecay 정의하기
        if self.current_ep > self.lrdecay_every + self.last_lrdecay and len(self.loss_list) > 2.5*self.lrdecay_window:

            loss_array = np.asarray(self.loss_list)
            before_loss = loss_array[loss_array.size - self.lrdecay_window*2:loss_array.size-self.lrdecay_window]
            current_loss = loss_array[loss_array.size - self.lrdecay_window:loss_array.size]
            before_average = np.average(before_loss)
            current_average = np.average(current_loss)

            current_decay = (current_loss - before_average)/before_average
            decay_mask = current_decay < -self.lrdecay_thres
            not_decay = np.sum(decay_mask) < 0.1*self.lrdecay_window


            current_delta = np.asarray([current_loss[i] - current_loss[i-1] for i in range(1,len(current_loss))])
            current_delta = np.abs(current_delta)/current_average
            current_mask = current_delta>self.lrdecay_thres

            oscilliate = np.sum(current_mask) > 0.5*self.lrdecay_window

            if not_decay or oscilliate:
                print(decay_mask.shape)
                print(current_mask.shape)
                self._lrdeday()
                return True
            else:
                return False
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
if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('./101_swallow_0.5_all_ep1150.csv')
    vl = df['t_loss']
    vl = np.asarray(vl)
    print(len(vl))
    window = 50
    lrdecay_window = window
    lrdecay_thres = 0.1


    for current_ep in range(len(vl)):
        loss_list = vl[:current_ep]
        if len(loss_list) > 2.5 * window:
            loss_array = np.asarray(loss_list)
            before_loss = loss_array[current_ep - lrdecay_window * 2:current_ep - lrdecay_window]
            current_loss = loss_array[current_ep - lrdecay_window:current_ep]
            before_average = np.average(before_loss)
            current_average = np.average(current_loss)

            current_decay = (current_loss - before_average) / before_average
            decay_mask = current_decay < -lrdecay_thres
            not_decay = np.sum(decay_mask) < 0.1* lrdecay_window

            current_delta = np.asarray([current_loss[i] - current_loss[i - 1] for i in range(1, len(current_loss))])
            current_delta = np.abs(current_delta) / current_average
            current_mask = current_delta > lrdecay_thres

            oscilliate = np.sum(current_mask) > 0.5* lrdecay_window

            if oscilliate:
                print('{} oscilliate'.format(current_ep, oscilliate))

            if not_decay:
                print('{} notdecay'.format(current_ep, not_decay))


