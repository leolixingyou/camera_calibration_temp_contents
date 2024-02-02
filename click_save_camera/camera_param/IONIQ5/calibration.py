import os
import cv2
import numpy as np

class Calibration:
    def __init__(self, path):
        # camera parameters
        f60= path
        cam_param_f60 = []
        with open(f60, 'r') as f:
            data = f.readlines()
            for content in data:
                content_str = content.split()
                for compo in content_str:
                    cam_param_f60.append(float(compo))
        self.camera_matrix = np.array([[cam_param_f60[0], cam_param_f60[1], cam_param_f60[2]], 
                                       [cam_param_f60[3], cam_param_f60[4], cam_param_f60[5]], 
                                       [cam_param_f60[6], cam_param_f60[7], cam_param_f60[8]]])
        self.dist_coeffs = np.array([[cam_param_f60[9]], [cam_param_f60[10]], [cam_param_f60[11]], [cam_param_f60[12]]])




if __name__ == "__main__":

    camera_path = [
                './calibration_data/f60.txt',
                './calibration_data/f120.txt',
                './calibration_data/r120.txt'
                ]
    cal = Calibration(camera_path)
    front_img = cv2.undistort(front_img, self.calib.camera_matrix_f60, self.calib.dist_coeffs__f60, None, self.calib.camera_matrix_f60)
