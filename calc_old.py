import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.io as spio

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
    return np.arccos(cosangle)

####COBBANGLE 계산하는거 틀린거같음. 이거 의미를 찾자.
#그전까지 메인제외 o return
def calc_angle(coord,image_size , vnum = 7):
    H,W = image_size

    coord = coord.reshape(-1, 2)  # if the coord is linear

    #midlines : the middle line per bone
    mid_lines = np.zeros((int(coord.shape[0] / 2), 2))

    for i in range(mid_lines.shape[0]):
        if i % 2 == 0:
            mid_lines[i] = (coord[i * 2] + coord[i * 2 + 2]) / 2.0
        else:
            mid_lines[i] = (coord[i * 2 - 1] + coord[2 * i + 1]) / 2.0

    #midpoints : the middle of left, right
    midlist = [(coord[2 * n] + coord[2 * n + 1]) / 2.0 for n in range(int(coord.shape[0] / 2))]
    mid_points = np.asarray(midlist)

    # the slopes of each bones
    vec_lines = [mid_lines[2 * i + 1] - mid_lines[2 * i] for i in range(int(mid_lines.shape[0] / 2))]
    vec_lines = np.asarray(vec_lines)

    angles = _get_angle(vec_lines, vec_lines)
    for i in range(angles.shape[0]): #adjusting arccos error
        angles[i][i] = 0

    pos = np.unravel_index(np.argmax(angles), shape=angles.shape)
    pt = angles[pos]
    pt = pt / np.pi * 180

    # Pos1 is lower by the rule of MATLAB:pos2 is the name of variable for matlab
    if pos[0] > pos[1]:
        pos2, pos1 = pos[0], pos[1]
    else:
        pos2, pos1 = pos[1], pos[0]

    ######

    # if isS(mid_points) == False:
    #     pos11 = 0
    #     pos22 = vnum +1
    #     mt = _get_angle(vec_lines[0], vec_lines[pos1])
    #     tl = _get_angle(vec_lines[vnum + 1], vec_lines[pos2])
    #     tl = tl / np.pi * 180
    #     mt = mt / np.pi * 180
    # else:
    #     if mid_points[(pos1 + 1) * 2 - 1, 1] + mid_points[(pos2 + 1) * 2 - 1, 1] < W:  ##이게 뭘 뜻하는거지?
    #         # Calculate the Cobb angle(Upside)
    #         angles1 = _get_angle(vec_lines[:pos1 + 1], vec_lines[pos1])
    #         pos11 = np.argmax(angles1)
    #         mt = angles1[pos11] / np.pi * 180
    #
    #         # Calculate the Cobb angle (downside)
    #         angles2 = _get_angle(vec_lines[pos2:vnum + 1], vec_lines[pos2])
    #         pos22 = np.argmax(angles2)
    #         tl = angles2[pos22] / np.pi * 180
    #
    #
    #     else:
    #         # Calculate the Cobb angle (upside)
    #         angles1 = _get_angle(vec_lines[:pos1 + 1], vec_lines[pos1])
    #         pos11 = np.argmax(angles1)
    #         mt = angles1[pos11] * 180 / np.pi
    #
    #         # Calculate the Cobb angle (Upper upside)
    #         angles2 = _get_angle(vec_lines[:pos11 + 1], vec_lines[pos11])
    #         pos22 = np.argmax(angles2)
    #         tl = angles2[pos22] * 180 / np.pi
    mt = 0
    tl = 0
    pos11 = 0
    pos22 = 0
    cobb_angles = (pt, mt, tl)
    pos = (pos1, pos2, pos11, pos22)
    return cobb_angles,pos, mid_points, mid_lines, vec_lines

def _make_plot(title, image, mid_lines, pos, save_path = None, coord_list = None):
    plt.title(title)
    plt.imshow(image, cmap='gray')

    # Plotting the points
    plt.scatter(mid_lines[:, 0], mid_lines[:, 1], s=1.2, c='cyan')

    colors = ['magenta', 'green', 'yellow','blue']

    if coord_list is not None:
        for i in range(len(coord_list)):
            clr = colors[i]
            label = coord_list[i]
            label = label.reshape(-1, 2)
            plt.scatter(label[:, 0], label[:, 1], s=1.2, c=clr)

    pos = list(pos)
    pos.sort()#ascending

    for pos_value in pos:
        plt.plot(mid_lines[pos_value * 2:pos_value * 2 + 2, 0],
                 mid_lines[pos_value * 2:pos_value * 2 + 2, 1],
                 'c.-', alpha=0.5, linewidth=2)

    #plt.plot(mid_lines[pos2 * 2:pos2 * 2 + 2, 0], mid_lines[pos2 * 2:pos2 * 2 + 2, 1], 'c.-', alpha=0.5, linewidth=2)

    if save_path != None:
        plt.savefig(save_path)

def _plot_with_calc(image, label, title = None, label_ref = None, save_path = None):
    if len(image.shape) ==3:
        _,H,W = image.shape
    else:
        H,W = image.shape
    coord = label.reshape(-1,2)
    coord[:,0] *= W
    coord[:,1] *= H

    cobb_angles, pos, mid_points, mid_lines, vec_lines = calc_angle(coord, (H,W))
    if title is None:
        title = 'ang{}, pos{}'.format(cobb_angles, pos)
    if label_ref is None:
        landmark_list = None
    else:
        label_ref = label_ref.reshape(-1,2)
        coord_ref = np.zeros(label_ref.shape)
        coord_ref[:,0] = label_ref[:,0] * W
        coord_ref[:,1] = label_ref[:,1] * H
        landmark_list=  [coord, coord_ref]

    image = image.transpose((1,2,0))
    _make_plot(title, image, mid_lines, pos, coord_list= landmark_list, save_path = save_path)
    return cobb_angles, pos, mid_points, mid_lines, vec_lines