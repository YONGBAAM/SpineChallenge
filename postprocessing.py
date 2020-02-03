import numpy as np
import os
from label_io import read_data_names, read_labels, chw, hwc, write_labels
import matplotlib.pyplot as plt
import pandas as pd
from label_io import to_absolute, to_relative, plot_image, label_sort, read_images
from calc import calc_angle_old, _get_angle, calc_angle, get_mid_points, get_vec_lines
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import time

#############################################
#
#   잘못됬음!! label가지고 각도 구할때는 원본이미지 스케일에서 해야 함!
#   테스트 이미지 이터레이터 구현
#   이제 노패드니까 프레딕트도 relative label로 걍 두기
#
#
#   정리는 더 안해도 된다.
###############################################
def calc_poli(x, coeffs):
    res = np.zeros_like(x)
    for coeff in coeffs:
        res *= x
        res = res + coeff
    return res

def label_fit(label,degree, full = False):
    label = label.reshape(-1,2)
    coeff = np.polyfit(x=label[:, 1], y=label[:, 0], deg=degree)
    out_xaxis = calc_poli(label[:,1], coeff)
    out_label = np.concatenate((out_xaxis.reshape(-1,1), label[:,1].reshape(-1,1)), axis = 1)
    out_label = out_label.reshape(-1)
    if full:
        return out_label, coeff
    else:
        return out_label

def derivative_poli(coeff):
    if len(coeff) ==1:
        return [0]
    else:
        N = len(coeff)
        der = []
        degree = N-1
        for ind in range(N-1):
            dc = degree * coeff[ind]
            der.append(dc)
            degree -=1
    return der

from sklearn.metrics import r2_score
def determine_degree(true_labels):
    true_labels = true_labels.reshape(-1,34,2,2).copy()
    result_list = []
    for fit_degree in range(3,9):
        r2 = 0
        for label in true_labels:
            left = label[:, 0, :]
            left_fit, coeff_l = label_fit(left, fit_degree, full=True)
            left_fit_x = left_fit.reshape(-1,2)[:,0]
            left_ori_x = left.reshape(-1,2)[:,0]
            r2_l = r2_score(left_fit_x, left_ori_x)

            right = label[:, 1, :]
            right_fit, coeff_r = label_fit(right, fit_degree, full=True)
            right_fit_x = right_fit.reshape(-1,2)[:,0]
            right_ori_x = right.reshape(-1, 2)[:, 0]
            r2_r = r2_score(right_fit_x, right_ori_x)
            r2 += r2_l + r2_r
        result_list.append(dict(deg = fit_degree, r2 = r2))
    print(result_list)


#with predicted coordinate
#integrating
def post_way1(label_pred_abs, degree = 6, full = False):
    #pred_path = './model/RH_SM_all_ep1999'

    ####    Get ablsolute prediction
    fit_degree = degree
    pred = label_pred_abs
    pred = pred.reshape(34, 2, 2)

    ########################################################
    #   Left fit과 right fit 만들고 왼쪽 오른쪽 각각에 대해
    #   landmark 구하기
    #   해당 랜드마크 가지고 angle prediction
    ########################################################

    left = pred[:, 0, :]
    left_fit, coeff_l = label_fit(left, fit_degree, full=True)

    right = pred[:, 1, :]
    right_fit, coeff_r = label_fit(right, fit_degree, full=True)

    fitted_preds = np.concatenate((left_fit.reshape(-1, 1, 2), right_fit.reshape(-1, 1, 2))
                                  , axis=1)
    fitted_preds = fitted_preds.flatten().astype(int)

    # fit line 추가
    d = 1
    pred = pred.reshape(34, 2, 2)
    minyl = pred[0, 0, 1]
    minyr = pred[0, 1, 1]
    maxyl = pred[33, 0, 1]
    maxyr = pred[33, 1, 1]
    left_r = np.arange(minyl, maxyl, d)
    right_r = np.arange(minyr, maxyr, d)
    left_c = calc_poli(left_r, coeff_l)
    right_c = calc_poli(right_r, coeff_r)
    left_line = np.concatenate((left_c.reshape(-1, 1), left_r.reshape(-1, 1)), axis=1)
    right_line = np.concatenate((right_c.reshape(-1, 1), right_r.reshape(-1, 1)), axis=1)


    params = dict(
        left_coeff = coeff_l, right_coeff = coeff_r,
        left_line = left_line, right_line = right_line

    )
    if full:
        return fitted_preds, params
    else:
        return fitted_preds

def post_way2(label_pred_abs, degree = 6, full = False):

    # pred_path = './model/RH_SM_all_ep1999'

    ####    Get ablsolute prediction
    fit_degree = degree
    pred = label_pred_abs

    pred = pred.reshape(34, 2, 2)

    left = pred[:, 0, :]
    left_fit, coeff_l = label_fit(left, fit_degree, full=True)
    right = pred[:, 1, :]
    right_fit, coeff_r = label_fit(right, fit_degree, full=True)

    fitted_preds = np.concatenate((left_fit.reshape(-1, 1, 2), right_fit.reshape(-1, 1, 2))
                                  , axis=1)
    fitted_preds = fitted_preds.flatten().astype(int)

    vec_with_fitted_preds = get_vec_lines(fitted_preds)

    ideal_slope_top = vec_with_fitted_preds[0]
    tan_top = np.asarray([ideal_slope_top[1], -ideal_slope_top[0]])
    tan_top /= np.linalg.norm(tan_top)

    ideal_slope_bot = vec_with_fitted_preds[-1]
    tan_bot = np.asarray([ideal_slope_bot[1], -ideal_slope_bot[0]])
    tan_bot /= np.linalg.norm(tan_bot)

    pred = pred.reshape(17, 4, 2)
    midpoints4 = np.zeros((17, 2))
    for i, pnts in enumerate(pred):
        midpoints4[i] = np.average(pnts, axis=0)
    del_midpoint_y = np.average([midpoints4[i+1,1] - midpoints4[i,1] for i in range(16)])

    top_pad = pred[0] + 1*del_midpoint_y*tan_top
    bot_pad = pred[-1] - 1*del_midpoint_y * tan_bot




    #이전방법 깃헙에 백업해놓기

    #for middle fit

    # coeff = list((coeff_l + coeff_r) / 2)


    #중점으로 플롯 0.56
    # mid = (left + right) /2
    # _,coeff = label_fit(mid, fit_degree, full = True)

    #모든좌표로 플롯
    conc = np.concatenate((top_pad.reshape(-1,2), left, right, bot_pad.reshape(-1,2)), axis = 0)
    _,coeff = label_fit(conc, fit_degree, full = True)

    #test
    # plt.figure()
    # plot_image(image = np.ones((2000,1200)), coord_red=conc, coord_gr=bot_pad, coord_bl=top_pad)
    # plt.show()

    der_coeff = derivative_poli(coeff)
    # 절대좌표니까 괜찮음상대좌표는 해줘야함

    # midpoint 추정을 피팅한걸로 해서 할 수도 있다
    ############################################
    #
    #   midpoint curve의 도함수를 구하고
    #   추정랜드마크의 r값을 통해 해당 기울기 구함
    #   그다음 그거가지고 angle구함
    #
    #############################################
    midpoints_x = midpoints4[:, 1]

    der_y = calc_poli(midpoints_x, der_coeff)
    # 1,-m이 slope vector
    slopes = np.ones((17, 2))
    slopes[:, 1] = -der_y

    #처음 끝 보정
    # slopes[0] = (vec_with_fitted_preds[0] + slopes[0])/2
    # slopes[-1] = (vec_with_fitted_preds[-1] + slopes[-1])/2



    ####    Ploting
    d = 1
    pred = pred.reshape(34, 2, 2)
    minyl = pred[0, 0, 1]
    minyr = pred[0, 1, 1]
    miny = int((minyl + minyr) / 2)
    maxyl = pred[33, 0, 1]
    maxyr = pred[33, 1, 1]
    maxy = int((maxyl + maxyr) / 2)
    middle_r = np.arange(miny, maxy, d)
    middle_c = calc_poli(middle_r, coeff)
    middle_line = np.concatenate((middle_c.reshape(-1, 1), middle_r.reshape(-1, 1)), axis=1)

    params = dict(
        coeff = coeff, line = middle_line, middle_vec = midpoints4
    )
    if full:
        return slopes, params
    else:
        return slopes

def postprocess_inte(pred_path, images, labels_gt_abs, title=None, save_plot=False, automatic=False,
                     automatic_time = 10, degree = 4):
    author = 'YB'
    #####   Get absolute pred
    if not os.path.exists(os.path.join(pred_path, 'labels_pred_abs.csv')):
        preds_rel = read_labels(pred_path, title='labels_pred_rel')
        preds = []
        for ind, pred_rel in enumerate(preds_rel):
            _img = images[ind]
            H, W, _ = _img.shape
            pred_abs = to_absolute(label=pred_rel, H=H, W=W)
            preds.append(pred_abs)
        preds = np.asarray(preds)
        write_labels(preds, label_location=pred_path, title='labels_pred_abs')
    else:
        preds = read_labels(pred_path, title='labels_pred_abs')
    preds = label_sort(preds)

    result_list = []
    for ind, image in enumerate(images):
        gt = labels_gt_abs[ind]
        pred = preds[ind]
        H, W, C = image.shape

        fitted_preds, params1 = post_way1(pred, full=True, degree = degree)
        slopes, params2 = post_way2(pred, full=True, degree = degree)

        gt_angles, gt_pos = calc_angle_old(gt, (H, W), full=True)
        w1_angles, w1_pos = calc_angle_old(fitted_preds, (H, W), full=True)
        w2_angles, w2_pos = calc_angle(slopes, get_mid_points(pred),image_H=H, full=True)

        #convert to smape error
        w1_error = np.sum(np.abs(w1_angles - gt_angles)) / np.sum(gt_angles + w1_angles)
        w2_error = np.sum(np.abs(w2_angles - gt_angles)) / np.sum(gt_angles + w2_angles)

        w1_mainerr = (np.abs(w1_angles - gt_angles) / (gt_angles + w1_angles + 1e-8))[0]
        w2_mainerr = (np.abs(w2_angles - gt_angles) / (gt_angles + w2_angles + 1e-8))[0]

        pred = pred.reshape(-1, 2, 2)
        delta = pred[:, 0, :] - pred[:, 1, :]
        deltax = np.average(np.abs(delta)[:, 0]) / W
        # show는 테스트
        if save_plot or automatic:

            #make plot
            plt.figure()
            # R for w1
            left_line = params1['left_line']
            right_line = params1['right_line']
            plt.plot(left_line[:, 0], left_line[:, 1], color='magenta', alpha=0.5)
            plt.plot(right_line[:, 0], right_line[:, 1], color='magenta', alpha=0.5)

            _fp = fitted_preds.reshape(-1,2,2).copy()
            for pos in w1_pos:
                dots = np.average(_fp[2*pos:2*pos+2,:,:], axis = 0)
                plt.plot(dots[:,0], dots[:,1], 'r')

            # blue for w2
            middle_line = params2['line']
            plt.plot(middle_line[:, 0], middle_line[:, 1], color='cyan', alpha=0.5)

            vec_center = params2['middle_vec']
            for pos in w2_pos:
                center = vec_center[pos]
                _slope = slopes[pos]
                norm_slope = _slope / np.linalg.norm(_slope)
                right = center + norm_slope * 50
                left = center - norm_slope * 50
                plt.plot([left[0], right[0]], [left[1], right[1]], color='blue')

            plot_image(image, coord_red=fitted_preds, coord_gr=gt)
            plt_title = 'a %.1f/p1 %.1f/err%% %.2f'%\
                        (gt_angles[0], w2_angles[0], w2_error*100)
            plt.title('test_{}\n'.format(ind) + plt_title)

            if automatic:
                plt.show()
                plt.pause(automatic_time)
                plt.close()
            if save_plot:
                if title is None:
                    title = 'pred'
                if not os.path.exists(os.path.join(pred_path, title)):
                    os.makedirs(os.path.join(pred_path, title))
                plt.savefig(os.path.join(pred_path,title, title + '_{}'.format(ind)+ '.png'))
                plt.close()

        res_dict = dict(gt_angle1=gt_angles[0], gt_angle2=gt_angles[1], gt_angle3=gt_angles[2],
                        w1_angle1=w1_angles[0], w1_angle2=w1_angles[1], w1_angle3=w1_angles[2],
                        w2_angle1=w2_angles[0], w2_angle2=w2_angles[1], w2_angle3=w2_angles[2],
                        w1_mainerr=w1_mainerr, w2_mainerr=w2_mainerr,
                        w1_error=w1_error,
                        w2_error=w2_error,
                        avgdx=deltax, gt_pos=gt_pos, w1_pos=w1_pos, w2_pos=w2_pos
                        )
        result_list.append(res_dict)
    w1_errors = np.array([di['w1_error'] for di in result_list])
    w2_errors = np.array([di['w2_error'] for di in result_list])

    print('Avg : w1_e %.2f / w2_e %.2f '%(np.average(w1_errors), np.average(w2_errors)))

    df = pd.DataFrame(result_list)
    df.to_csv(os.path.join(pred_path, 'dict_result_%.2f.csv'%(np.average(w2_errors))))

    return result_list

if __name__ == '__main__':
    test_label_location = './test_labels'
    test_data_location = './test_images'
    test_data_names = read_data_names(test_label_location)
    test_labels = read_labels(test_label_location)
    test_images = read_images(test_data_location, test_data_names)

    train_label_location = './train_labels'
    train_image_location = './train_images'

    train_labels = read_labels(train_label_location)

    #determine_degree(train_labels)

    #get slope distribution


    pred_path = './model/34_Fin_Grad_ep3986'
    plt.rcParams["figure.figsize"] = (4, 8)

    # for deg in range(3,10):
    #     print(deg)
    #     postprocess_inte(pred_path=pred_path, images=test_images,
    #                      labels_gt_abs=test_labels, title='Way_1_Fin', save_plot = False, automatic=False, degree=deg,
    #                      automatic_time=30)

    postprocess_inte(pred_path=pred_path, images=test_images,
                         labels_gt_abs=test_labels, title='Way_2_fin', save_plot = True, automatic=False, degree=6,
                         automatic_time=30)

    # postprocess_inte(pred_path = pred_path, images = test_images,
    #                  labels_gt_abs = test_labels, title=None, save_plot = True, automatic=False,
    #                      automatic_time= 30)
