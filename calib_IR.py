# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import glob
import pickle
import time
import multiprocessing
import matplotlib.pyplot as plt

def calib(inter_corner_shape, size_per_grid, img_dir,img_type, save_dir,CAMERA_PARAMETERS_FILE):
    
    w,h = inter_corner_shape
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    cp_world = cp_int*size_per_grid
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        # img = cv2.resize(img,(1280,720))
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w,h), None)
        print()
        # if ret is True, save.
        if ret == True:
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            # view the corners
            cv2.drawChessboardCorners(img, (w,h), cp_img, ret)
            # cv2.imshow('FoundCorners',img)
            cv2.imwrite(save_dir + os.sep + img_name, img)
            print(fname)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            cv2.waitKey(1)

    # calibrate the camera
    print("caculating...")

    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)

    # calculate the error of reproject
    total_error = 0
    error_list =[]
    x_error_list = []
    y_error_list = []
    x_error_ratio_list = []
    y_error_ratio_list = []
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        num_points = img_points[i]
        for j in range(len(num_points)):
            x_error = num_points[j][0][0] - img_points_repro[j][0][0]
            y_error = num_points[j][0][1] - img_points_repro[j][0][1]
            x_error_ratio = (num_points[j][0][0] - img_points_repro[j][0][0])/num_points[j][0][0]
            y_error_ratio = (num_points[j][0][1] - img_points_repro[j][0][1])/num_points[j][0][1]
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
        total_error += error
        error_list.append(total_error/(i+1))
        x_error_list.append(x_error)
        y_error_list.append(y_error)
        x_error_ratio_list.append(x_error_ratio)
        y_error_ratio_list.append(y_error_ratio)
    Average_error = total_error/len(obj_points)
    print(("Average Error of Reproject: "),Average_error )
    data = {"ret":ret,
            "mat_inter":mat_inter,
            "coff_dis":coff_dis,
            "v_rot":v_rot,
            "v_trans":v_trans,
            "Average_error":Average_error}
    

    '''txt'''
    for index, msg in enumerate(data.items()):
        key, value = msg
        if index == 0:
            with open(img_dir + CAMERA_PARAMETERS_FILE + ".txt", 'w') as f :
                f.write(str(key) + ':' + '\n' + str(value) + '\n')
        else:
            with open(img_dir + CAMERA_PARAMETERS_FILE + ".txt", 'a') as f :
                f.write(str(key) + ':' + '\n' + str(value) + '\n')
    return mat_inter, coff_dis
    
def dedistortion(inter_corner_shape, img_dir,img_type, save_dir, mat_inter, coff_dis):
    w,h = inter_corner_shape
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        img = cv2.resize(img,(1280,720))
        dst = cv2.undistort(img, mat_inter, coff_dis, None, mat_inter)
        cv2.imwrite(save_dir + os.sep + img_name, dst)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print('Dedistorted images have been saved to: %s successfully.' %save_dir)
    
def mkdir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

if __name__ == '__main__':
    ### original
    # inter_corner_shape = (8,6)
    # size_per_grid0 = 0.025

    ### zhang method data
    inter_corner_shape = (7,5)
    size_per_grid0 = 0.037

    img_type0 = "jpg"
    img_type1 = "png"

    path = "./Iphone 13 pro max/"

    img_dir2 = path
    name2 = 'iphone'
    save_pio2 = img_dir2 + "poi/"

    #calibrate the camera
    mkdir(save_pio2)
    mat_inter2, coff_dis2 = calib(inter_corner_shape, size_per_grid0, img_dir2,img_type0, save_pio2,name2)

    save_dir2 = img_dir2 + 'dis/'
    
    mkdir(save_dir2)

    dedistortion(inter_corner_shape, img_dir2, img_type0, save_dir2, mat_inter2, coff_dis2)



