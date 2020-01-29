import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.io as spio

path = './boostnet_labeldata'
training_data_path = path + '/data/training'
training_label_path = path + '/labels/training'

plot_save_path = './label_plots'

label_path_all = os.listdir(training_label_path)

label_list = []

output_list = []

ap_num = 32;
lat_num = 32;

show_every = 5000
save_mode = True
testMode = False

def isS(p):
    H = p.shape[0]
    ll = np.zeros((H - 2, 1))
    for i in range(H - 2):
        ll[i,0] = (p[i, 1] - p[H - 1, 1]) / (p[0, 1] - p[H - 1, 1]) - (p[i, 0] - p[H - 1, 0]) / (p[0, 0] - p[H - 1, 0])
    S = np.matmul(ll, ll.T)

    flag = S < 0
    return np.sum(flag) >= 1

for label_name in label_path_all:
    if '.mat' == label_name[-4:]:
        label_list.append(label_name)

data_list = [label_name[:-4] for label_name in label_list]

landmarks_ap = []
landmarks_lat = []

import pandas as pd

label_path_training = './boostnet_labeldata/labels/training/angles.csv'
true_label_df = pd.read_csv(label_path_training, index_col=None, header=None)
true_label_df.rename(columns={0: 'true0', 1: 'true1', 2: 'true2'}, inplace=True)






#####################@@@@@@@@@@@
for data_no in range(len(data_list)):
#for data_no in range(1):
    image =Image.open(training_data_path + '/' + data_list[data_no])
    image = np.asarray(image)
    lb = spio.loadmat(training_label_path + '/' + label_list[data_no])
    coord = np.array(lb['p2'])

    H,W = image.shape

    ##Landmark : normalized된 좌표 concatenating
    if 'lateral' in label_list[data_no]:
        vnum = int(lat_num/4) -1
        coord = coord[:lat_num*2]
        normalized_x_axis = coord[:, 0] / W
        normalized_y_axis = coord[:, 1] / H
        landmark = np.concatenate((normalized_x_axis, normalized_y_axis), axis = 0)
        landmarks_lat.append(landmark)
    else:
        vnum = int(ap_num/4) -1
        coord = coord[:ap_num*2]
        normalized_x_axis = coord[:, 0] / W
        normalized_y_axis = coord[:, 1] / H
        landmark = np.concatenate((normalized_x_axis, normalized_y_axis), axis = 0)
        landmarks_ap.append(landmark)

    cobb_angles, pos, mid_points, mid_lines, vec_lines = calc_angle(coord = coord, image_size = (H,W), vnum = vnum)

    pos1, pos2, pos11, pos22 = pos

    true_angle = true_label_df.iloc[data_no]
    error = [0,0,0]
    for i in range(3):
        if true_angle[i] != 0:
            error[i] = np.abs(cobb_angles[i] - true_angle[i]) / true_angle[i] * 100
    output_string_test = '%d , pos:%d %d, %.1f/%.1f/%.1f, error%% %.1f/%.1f/%.1f' % (
    data_no, pos1, pos2, cobb_angles[0], cobb_angles[1], cobb_angles[2],
    error[0], error[1], error[2])

    if testMode == True:
        print(output_string_test)
        continue


    # output_list.append({'no' : data_no,
    #                     'cobb_angles' : cobb_angles,
    #                     'pos1' : pos1,
    #                     'pos2' : pos2})

    output_list.append(np.array([cobb_angles[0], cobb_angles[1], cobb_angles[2],
                                  pos1, pos2]))

    save_path = plot_save_path + '/' + data_list[data_no] if save_mode == True else None

    fig = make_plot('angle : %.2f most tilted : %d, %d, u%d, l%d' % (cobb_angles[0], pos1, pos2, pos11, pos22),
                    image, mid_lines, pos)

    # plt.figure()
    # plt.title('angle : %.2f most tilted : %d, %d' % (pt, pos2, pos1))
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # # plt.plot(mid_lines[pos1,0], mid_lines[pos1,1])
    #
    # # Plotting the points
    # plt.scatter(mid_lines[:, 0], mid_lines[:, 1], s=1.2, c='yellow')
    #
    # # Calculating the cross point for two vectors.
    #
    # # v11 = mid_lines[pos1 * 2]
    # # v12 = mid_lines[pos1 * 2 + 1]
    # # v21 = mid_lines[pos2 * 2]
    # # v22 = mid_lines[pos2 * 2 + 1]
    # # s1 = v12 - v11  # slope
    # # s2 = v22 - v11
    # # t2 = (v11[0] - v21[0]) * s1[1] - (v11[1] - v21[1]) * s1[0]
    # # t2 /= (s2[0] * s1[1] - s2[1] * s1[0])
    # # t1 = (s2[1] * s1[1] * t2 - (v11[1] - v21[1]) * s1[0]) / (s1[0] * s1[1])
    # # cross_point = v11 + t1 * (s1)
    #
    # plt.plot()
    #
    # # plt.plot(cross_point[0], cross_point[1], v11[0], v11[1], 'c.-', alpha = 0.5,  linewidth = 2)
    # # plt.plot(cross_point[0], cross_point[1], v21[0], v21[1], 'c.-', alpha = 0.5,  linewidth = 2)
    # plt.plot(mid_lines[pos1 * 2:pos1 * 2 + 2, 0], mid_lines[pos1 * 2:pos1 * 2 + 2, 1], 'c.-', alpha=0.5, linewidth=2)
    # plt.plot(mid_lines[pos2 * 2:pos2 * 2 + 2, 0], mid_lines[pos2 * 2:pos2 * 2 + 2, 1], 'c.-', alpha=0.5, linewidth=2)

    if data_no % show_every ==0:
        plt.show()
        output_string = "{} : (PT, MT, TL/L) are {}, {}, {}\n " \
                        "and the two most tilted vertebrae are {}, {}".format(data_no, cobb_angles[0], cobb_angles[1], cobb_angles[2], pos2, pos1)
        print(output_string)
    plt.close()


#get label angle


# calculated_df = pd.DataFrame(np.array(output_list), columns = ['calc0', 'calc1', 'calc2', 'pos1', 'pos2'])
#
# calculated_df.insert(column =['true0'], value = true_label_df['true0'] )
# calculated_df.insert(column =['true1'], value = true_label_df['true1'] )
# calculated_df.insert(column =['true2'], value = true_label_df['true2'] )

##converting output list to pandas dataframe

'''
     fprintf(output);
    %         fprintf('No. %d :The Cobb Angles(PT, MT, TL/L) are %3.1f, and the two most tilted vertebrae are %d and %d. ',...
    %             k,CobbAn,pos2,pos1(pos2));
    
    pause(200)
    close all
    CobbAn_ap = []
    CobbAn_lat = []
    
    
        %isempty(strfind(lower(fileNames{k}),'lateral'))
        
    %AP인지 LAT인지에 따라 다른듯!!
    CobbAn_ap = [CobbAn_ap ; cob_angles]; %cobb angles
    
        
    CobbAn_lat = [CobbAn_lat ; cob_angles]; %cobb angles
    

end

% write to csv file
csvwrite('angles_ap.csv',CobbAn_ap);
csvwrite('angles_lat.csv',CobbAn_lat);
csvwrite('landmarks_ap.csv',landmarks_ap);
csvwrite('landmarks_lat.csv',landmarks_lat);

fid = fopen('filenames_aplat.csv','wt');

if fid>0
    for k=1:N
        fprintf(fid,'%s\n',fileNames_im{k});
    end
        fclose(fid);
end


    
    '''

