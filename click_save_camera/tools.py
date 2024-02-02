
import os
import cv2
import copy
import math
import random
import numpy as np


def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

def get_file_list(path, file_type):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in file_type:
                image_names.append(apath)
    return image_names

def vector_from_vp_to_cp(vp, cp):
    return cp - vp

def project_to_image(points, proj_mat):
    num_pts = points.shape[1]
    points = np.vstack((points, np.ones((1, num_pts))))
    points = np.dot(proj_mat, points)
    points[:2, :] /= points[2, :]
    return points[:2, :]