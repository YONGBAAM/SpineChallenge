import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.io as spio
from label_io import to_absolute, to_relative, plot_image, label_sort, read_images

###############
#
#       angle 구하는 알고리즘 이상함.
#       트레이닝 셋으로 실험해보기 알고리즘수정 (수정안한 데모데이터로)
#       일단 오늘은 패스하기
###################

def isS(p):
    H = p.shape[0]
    ll = np.zeros((H - 2, 1))
    for i in range(H - 2):
        ll[i,0] = (p[i, 1] - p[H - 1, 1]) / (p[0, 1] - p[H - 1, 1]) - (p[i, 0] - p[H - 1, 0]) / (p[0, 0] - p[H - 1, 0])
    S = np.matmul(ll, ll.T)

    flag = S < 0
    return np.sum(flag) >= 1

#N*2, M*2 -> N*M
def _get_angle(nv1, nv2):
    nv11 = nv1.reshape(-1, 2)
    nv22 = nv2.reshape(-1, 2)
    nrm1 = np.linalg.norm(nv11, axis = 1).reshape(-1, 1)
    nrm1 = np.tile(nrm1, (1, nv11.shape[1]))
    nrm2 = np.linalg.norm(nv22, axis = 1).reshape(-1, 1)
    nrm2 = np.tile(nrm2, (1, nv22.shape[1]))

    nv11 = nv11 / nrm1
    nv22 = nv22 / nrm2
    cosangle = np.round(np.matmul(nv11, nv22.T), 8)
    cosangle = np.clip(cosangle, -1, 1)
    return np.arccos(np.round(cosangle, 12))

def get_vec_lines(label):
    label = label.reshape(68, 2)  # if the coord is linear

    # midlines : the middle line per bone
    mid_lines = np.zeros((34, 2))

    for i in range(mid_lines.shape[0]):
        if i % 2 == 0:
            mid_lines[i] = (label[i * 2] + label[i * 2 + 2]) / 2.0
        else:
            mid_lines[i] = (label[i * 2 - 1] + label[2 * i + 1]) / 2.0
    vec_lines = np.zeros((17, 2))
    for i in range(17):
        vec_lines[i] = mid_lines[2 * i + 1] - mid_lines[2 * i]
    return vec_lines

def get_mid_points(label):
    label = label.reshape(68, 2)
    mid_points = np.zeros((34, 2))
    for n in range(34):
        mid_points[n] = (label[2 * n] + label[2 * n + 1]) / 2
    return mid_points

#calc angle from
def calc_angle(vec_lines, mid_points, image_H, full = False):
    H = image_H
    #midlines : the middle line per bone
    vec_lines = vec_lines.reshape(17,2)
    mid_points = mid_points.reshape(34,2)
    angles = _get_angle(vec_lines, vec_lines)

    pos = np.unravel_index(np.argmax(angles), shape=angles.shape)
    pt = angles[pos]
    pt = pt / np.pi * 180
    # Pos1 is lower by the rule of MATLAB:pos2 is the name of variable for matlab
    if pos[0] > pos[1]:
        pos2, pos1 = pos[0], pos[1]
    else:
        pos2, pos1 = pos[1], pos[0]


    if not isS(mid_points):
        case = 0
        pos11 = -1
        pos22 = -1
        mt = _get_angle(vec_lines[0], vec_lines[pos1]).item()
        tl = _get_angle(vec_lines[16], vec_lines[pos2]).item()
        if type(mt) == type(np.zeros((1,1))):
            mt = mt.item()
        if type(tl) == type(np.zeros((1,1))):
            tl = tl.item()
        tl = tl / np.pi * 180
        mt = mt / np.pi * 180
    else:
        if mid_points[(pos1 + 1) * 2 - 1, 1] + mid_points[(pos2 + 1) * 2 - 1, 1] < H:  ##이게 뭘 뜻하는거지?
            case = 1
            # Calculate the Cobb angle(Upside)
            angles1 = _get_angle(vec_lines[:pos1 + 1], vec_lines[pos1])
            pos11 = np.argmax(angles1)
            _pos11 = np.unravel_index(pos11, angles1.shape)
            mt = angles1[_pos11] / np.pi * 180

            # Calculate the Cobb angle (downside)
            angles2 = _get_angle(vec_lines[pos2:17], vec_lines[pos2])
            pos22 = np.argmax(angles2)
            _pos22 = np.unravel_index(pos22, angles2.shape)
            pos22 = pos22 + pos2
            tl = angles2[_pos22] / np.pi * 180


        else:
            case = 2
            # Calculate the Cobb angle (upside)
            angles1 = _get_angle(vec_lines[:pos1 + 1], vec_lines[pos1])
            pos11 = np.argmax(angles1)
            _pos11 = np.unravel_index(pos11, angles1.shape)
            mt = angles1[_pos11] * 180 / np.pi

            # Calculate the Cobb angle (Upper upside)
            angles2 = _get_angle(vec_lines[:pos11 + 1], vec_lines[pos11])
            pos22 = np.argmax(angles2)
            _pos22 = np.unravel_index(pos22, angles2.shape)
            tl = angles2[_pos22] * 180 / np.pi
    cobb_angles = np.asarray([pt, mt, tl])
    pos = (pos1, pos2, pos11, pos22)

    #params = dict(pos = pos, mid_points = mid_points, vec_lines = vec_lines, case = case)
    if full:
        return cobb_angles, pos
    else:
        return cobb_angles

####COBBANGLE 계산하는거 틀린거같음. 이거 의미를 찾자.
#그전까지 메인제외 o return
def calc_angle_old(label, image_size, full = False):
    H = image_size[0]
    #Hardcode vnum


    label = label.reshape(68, 2)  # if the coord is linear

    #midlines : the middle line per bone
    mid_lines = np.zeros((34, 2))

    for i in range(mid_lines.shape[0]):
        if i % 2 == 0:
            mid_lines[i] = (label[i * 2] + label[i * 2 + 2]) / 2.0
        else:
            mid_lines[i] = (label[i * 2 - 1] + label[2 * i + 1]) / 2.0

    #midpoints : the middle of left, right
    mid_points = np.zeros((34,2))
    for n in range(34):
        mid_points[n] = (label[2 * n] + label[2 * n + 1]) / 2


    # midlist = [(labels[2 * n] + labels[2 * n + 1]) / 2.0 for n in range(int(labels.shape[0] / 2))]
    # mid_points = np.asarray(midlist)

    # the slopes of each bones
    vec_lines = np.zeros((17,2))
    for i in range(17):
        vec_lines[i] = mid_lines[2*i+1] - mid_lines[2*i]
    # vec_lines = [mid_lines[2 * i + 1] - mid_lines[2 * i] for i in range(int(mid_lines.shape[0] / 2))]
    # vec_lines = np.asarray(vec_lines)

    angles = _get_angle(vec_lines, vec_lines)

    pos = np.unravel_index(np.argmax(angles), shape=angles.shape)
    pt = angles[pos]
    pt = pt / np.pi * 180

    # Pos1 is lower by the rule of MATLAB:pos2 is the name of variable for matlab
    if pos[0] > pos[1]:
        pos2, pos1 = pos[0], pos[1]
    else:
        pos2, pos1 = pos[1], pos[0]

    ######

    if not isS(mid_points):
        case = 0
        pos11 = -1
        pos22 = -1
        mt = _get_angle(vec_lines[0], vec_lines[pos1]).item()
        tl = _get_angle(vec_lines[16], vec_lines[pos2]).item()
        if type(mt) == type(np.zeros((1,1))):
            mt = mt.item()
        if type(tl) == type(np.zeros((1,1))):
            tl = tl.item()
        tl = tl / np.pi * 180
        mt = mt / np.pi * 180
    else:
        if mid_points[(pos1 + 1) * 2 - 1, 1] + mid_points[(pos2 + 1) * 2 - 1, 1] < H:  ##이게 뭘 뜻하는거지?
            case = 1
            # Calculate the Cobb angle(Upside)
            angles1 = _get_angle(vec_lines[:pos1 + 1], vec_lines[pos1])
            pos11 = np.argmax(angles1)
            _pos11 = np.unravel_index(pos11, angles1.shape)
            mt = angles1[_pos11] / np.pi * 180

            # Calculate the Cobb angle (downside)
            angles2 = _get_angle(vec_lines[pos2:17], vec_lines[pos2])
            pos22 = np.argmax(angles2)
            _pos22 = np.unravel_index(pos22, angles2.shape)
            pos22 = pos22 + pos2
            tl = angles2[_pos22] / np.pi * 180


        else:
            case = 2
            # Calculate the Cobb angle (upside)
            angles1 = _get_angle(vec_lines[:pos1 + 1], vec_lines[pos1])
            pos11 = np.argmax(angles1)
            _pos11 = np.unravel_index(pos11, angles1.shape)
            mt = angles1[_pos11] * 180 / np.pi

            # Calculate the Cobb angle (Upper upside)
            angles2 = _get_angle(vec_lines[:pos11 + 1], vec_lines[pos11])
            pos22 = np.argmax(angles2)
            _pos22 = np.unravel_index(pos22, angles2.shape)
            tl = angles2[_pos22] * 180 / np.pi
    cobb_angles = np.asarray([pt, mt, tl])
    pos = (pos1, pos2, pos11, pos22)
    #params = dict(pos = pos, mid_points = mid_points, vec_lines = vec_lines, case = case)
    if full:
        return cobb_angles, pos
    else:
        return cobb_angles

from label_io import read_labels, read_images, read_data_names
import pandas as pd
if __name__ == '__main__':
    ####    Testing demo algorithm


    data_names_train = read_data_names('./train_labels')
    labels_now = read_labels('./train_labels')
    labels_ori = read_labels('./train_labels', title ='labels_original')
    labels_m = read_labels('./train_labels', title ='labels_m')
    train_images_location = './train_images'
    train_images = read_images(train_images_location, data_names=data_names_train)

    images = read_images('./train_images', data_names_train)

    # for ind, im in enumerate(images):
    #     lab_m = labels_m[ind]
    #     lab_ori = labels_ori[ind]
    #
    #     if np.sum(np.abs((lab_m - lab_ori)))>=1:
    #         plt.figure()
    #         plot_image(im, coord_red=lab_ori, coord_gr=lab_m)
    #         plt.show()

    #
    ####    gt angle
    gtangle = pd.read_csv('./train_labels/angles.csv', header=None, index_col=None)
    report_list = []
    allvecs = []
    editind = []
    for ind, img in enumerate(train_images):
        cobb_angles_gt = np.asarray(gtangle.iloc[ind, :])

        #label = labels_train[ind]
        label = labels_m[ind]
        H,W,_ = img.shape
        cobb_angles, pos, mid_points, mid_lines, vec_lines,case = calc_angle_old(label, image_size=(H, W))
        cobb_angles = np.squeeze(cobb_angles)
        cobb_angles = np.asarray([cobb_angles[0],cobb_angles[1], cobb_angles[2]]).flatten()
        err = np.abs(cobb_angles_gt - cobb_angles)/cobb_angles_gt
        err = err*100
        allvecs.append(np.concatenate((vec_lines[:,0], vec_lines[:,1]), axis = 0))
        report_list.append(dict(
            case = case,
            gt0 = cobb_angles_gt[0], gt1 = cobb_angles_gt[1], gt2 = cobb_angles_gt[2],
            ang0 = cobb_angles[0], ang1 = cobb_angles[1], ang2 = cobb_angles[2],
            err0 = err[0], err1 = err[1], err2 = err[2],
            my1 = pos[0]+1, my2 = pos[1]+1, my11 = pos[2]+1, my22 = pos[3]+1


        ))

    print(editind)
    df = pd.DataFrame(report_list)
    title = 'angle_calc_report_nosort'
    while os.path.exists(os.path.join('./train_labels', title + '.csv')):
        title += '_n'
    df.to_csv(os.path.join('./train_labels', title + '.csv'))
    print('SAVE : {}'.format(os.path.join('./train_labels', title + '.csv')))
    allvecs = np.asarray(allvecs)
    df = pd.DataFrame(allvecs)
    df.to_csv('./train_labels/veclines.csv', index = False, header= False)