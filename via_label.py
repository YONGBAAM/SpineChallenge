import os
from os.path import join as osj
import pandas as pd
from initial_processing import draw_seg
from label_io import label_sort_x, label_sort_y, label_sort, write_labels, plot_image
from PIL import Image
from label_io import hwc, chw
import numpy as np
import matplotlib.pyplot as plt
from label_io import read_data_names, write_labels, read_labels

plt.rcParams["figure.figsize"] = (6,12)
plt.ion()
def offset():
    # save offset
    ori_labels = read_labels('./record_labels')
    cr_labesl = read_labels('./record_cr_labels_v2')

    offset = ori_labels - cr_labesl

    from label_io import write_labels
    write_labels(offset, './', title='offset_v2_from_ori')


def what():
    # plt.ion()

    data_names = read_data_names('./record_labels')
    labels = read_labels('./record_labels')

    ori_location = './record_images'
    lr_location = './Old/record_cr_images_left'

    labels_cr = []
    for i, n in enumerate(data_names):
        txt_names = n.split('.')[0] + '.txt'
        label_ori = labels[i].reshape(-1, 2)
        im_ori = Image.open(osj(ori_location, n))
        im_ori = np.asarray(im_ori)

        im_lr = Image.open(osj(lr_location, n))
        im_lr = np.asarray(im_lr)

        Hori, Wori = im_ori.shape[0], im_ori.shape[1]
        Hlr, Wlr = im_lr.shape[0], im_lr.shape[1]

        topcrop = Hori - Hlr
        leftcrop = Wori - Wlr

        label_crop = np.zeros((68, 2))
        label_crop[:, 0] = label_ori[:, 0] - leftcrop
        label_crop[:, 1] = label_ori[:, 1] - topcrop
        labels_cr.append(label_crop.flatten())

    labels_cr = np.array(labels_cr)
    write_labels(labels_cr, './record_cr_labels_v2')

def record_crop():
    data_names = read_data_names('./record_labels')
    labels = read_labels('./record_labels')

    ori_location = './record_images'
    lr_location = './record_cr_images_left'
    cr_location = './record_cr_images'

    #leftright만 자르기

    labels_cr = []
    for i,n in enumerate(data_names):
        txt_names = n.split('.')[0] + '.txt'
        label_ori = labels[i].reshape(-1,2)
        im_ori = Image.open(osj(ori_location, n))
        im_ori = np.asarray(im_ori)

        im_lr = Image.open(osj(lr_location, n))
        im_lr = np.asarray(im_lr)

        im_cr = Image.open(osj(cr_location, n))
        im_cr = np.asarray(im_cr)

        Hori,Wori = im_ori.shape[0],im_ori.shape[1]
        Hlr, Wlr = im_lr.shape[0],im_lr.shape[1]
        Hcr, Wcr = im_cr.shape[0], im_cr.shape[1]

        topcrop = Hori-Hlr
        botcrop = Hlr-Hcr

        rightcrop_max = Wlr-Wcr
        leftcrop_max = Wori-Wlr

        #최소 0.05만큼은 자르기
        lc = int(0.05*Wori)
        rc = int(0.05*Wori)




        #아니면 현재에서 두배가 되도록 자르기

        lc = max(lc,leftcrop_max - int(Wcr/2))
        rc = max(rc,rightcrop_max - int(Wcr/2))

        #비율이 3:1이하면 자르지 말기
        if Hcr/Wcr<3:
            leftcrop = leftcrop_max
            rightcrop = rightcrop_max
        else:
            leftcrop = min(leftcrop_max, lc)
            rightcrop = min(rightcrop_max, rc)


        label_crop = np.zeros((68,2))
        label_crop[:,0] = label_ori[:,0] - leftcrop
        label_crop[:,1] = label_ori[:,1] - topcrop

        #save cropped image in grayscale
        cr_arr = im_ori[topcrop:Hori-botcrop,leftcrop:Wori-rightcrop]
        if len(cr_arr.shape) ==3:
            cr_arr = np.average(cr_arr, axis = 2)
        cr_im = Image.fromarray(cr_arr)
        cr_im  = cr_im.convert('L')
        cr_im.save(osj('record_cr_images_final', n))

        #im.average(axis = 2).axtype(int)

        with open(osj('./record_cr_labels', txt_names), 'w') as f:
            f.write('LRTB\n{},{},{},{}'.format(leftcrop,rightcrop, topcrop, botcrop))

        labels_cr.append(label_crop.flatten())
    labels_cr = np.array(labels_cr)
    write_labels(labels_cr, './record_cr_labels')

def label_import(name):
    plt.ion()

    df = pd.read_csv(name)

    print(df.iloc[0])

    coord_dict = {}

    import re
    xre = re.compile('"cx":([0-9]+)')
    yre = re.compile('"cy":([0-9]+)')

    for i in range(len(df)):
        d = df.iloc[i]
        filename = d['#filename']
        shape = d['region_shape_attributes']
        xcoord = xre.findall(shape)[0]
        ycoord = yre.findall(shape)[0]
        if filename not in coord_dict.keys():
            coord_dict[filename] = []

        coord_dict[filename].append([int(xcoord), int(ycoord)])

    npy_list = [n for n in os.listdir('./record_images_indexed') if n[-4:] == '.npy']
    seg_dict = {}
    for n, l in coord_dict.items():
        l = np.asarray(l).flatten()
        l = label_sort_x(l)
        l = label_sort_y(l)
        l = label_sort(l)

        number_char = '{}'.format(n.split('.')[0])

        if '{}.npy'.format(number_char) in npy_list:
            print('{} label already exist'.format(number_char))
            continue

        image = Image.open(osj('./record_images_indexed', n))
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = hwc(image)
            # print(image.shape)
        H, W, _ = image.shape

        # plt
        s = draw_seg(l, H, W)
        seg_dict[n] = s

    npy_list = [n for n in os.listdir('./record_images_indexed') if n[-4:] == '.npy']
    print('below are not saved')
    for n in coord_dict.keys():
        number_char = '{}'.format(n.split('.')[0])
        if '{}.npy'.format(number_char) not in npy_list:
            print(n)
    print('----------------------------')

    fig = plt.figure()
    plt.draw()
    plt.clf()
    plt.pause(0.1)

    for n, l in coord_dict.items():
        l = np.asarray(l).flatten()
        if l.size != 17 * 4 * 2:
            print('{} size is not 136, it is {}'.format(n, l.size))
            continue

        l = label_sort_x(l)
        l = label_sort_y(l)

        l = label_sort(l)

        number_char = '{}'.format(n.split('.')[0])
        if '{}.npy'.format(number_char) in npy_list:
            print('{} label already exist'.format(number_char))
            continue

        image = Image.open(osj('./record_images_indexed', n))
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = hwc(image)
            # print(image.shape)
        H, W, _ = image.shape

        # plt
        fig.suptitle(number_char)
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax.imshow(image)
        l = l.reshape(-1, 2)
        ax.scatter(l[:, 0], l[:, 1], s=10)
        seg = seg_dict[n]
        ax2.imshow(seg)
        plt.draw()
        plt.savefig('./rec_plots/{}.png'.format(n))
        plt.pause(0.1)

        while True:
            a = input()
            if a == 's':
                print('saving {}'.format(number_char))
                np.save(osj('./record_images_indexed', '{}.npy'.format(number_char)), l)
                break
            elif a == 'p':
                print('pass {}'.format(number_char))
                break
        plt.clf()
        plt.pause(0.1)

if __name__ == '__main__':
    data_names = read_data_names('./record_labels')
    ori_labels = read_labels('./record_labels')

    new_labels = []
    for i,n in enumerate(data_names):
        n = n.split('.')[0]
        txt_name = n + '.txt'
        with open(osj('./record_cr_labels', txt_name)) as f:
            text = f.read()
        cut = text.split('\n')[-1].split(',')
        print(cut)
        leftcut = int(cut[0])
        topcut = int(cut[2])

        label_ori = ori_labels[i].reshape(-1,2)
        new_label = np.zeros((68,2))
        new_label[:,0] = label_ori[:,0] - leftcut
        new_label[:,1] = label_ori[:,1] - topcut
        new_labels.append(new_label.flatten())
    write_labels(np.array(new_labels), './record_cr_labels')


    # labels = []
    # for i,n in enumerate(data_names):
    #     npyname = '{}'.format(100000+i)
    #     npyname = npyname[-4:] + '.npy'
    #     l = np.load(osj('./record_images_indexed',npyname))
    #     labels.append(l.flatten())
    # labels = np.array(labels)
    # write_labels(labels, './record_labels')
    #
    # offset = read_labels('./', 'offset_v2_from_ori')
    # v2_labels = labels - offset
    # write_labels(v2_labels,'./record_cr_labels_v2')
    # # label_import('./via_region_data (2).csv')
