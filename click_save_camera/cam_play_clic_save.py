#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import CompressedImage,Image
from sensor_msgs.msg import PointCloud2
import cv2
import time
from cv_bridge import CvBridge
from camera_param.IONIQ5.calibration import Calibration
from tools import *

FOLDER_NAME = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

class Image_click:
    def __init__(self, calib_dir, save_dir, img_size, save_file_name):
        self.calib = Calibration(calib_dir)
        # self.Unidistortion_mapping(self.calib,img_size)

        self.g_u = 0
        self.g_v = 0
        self.img_msg = None
        self.flag = False
        self.img_size = img_size
        self.save_dir = save_dir
        self.save_file_name = save_file_name
        self.temp_coords = []
        self.zoom_level = 1.0  # 初始缩放级别为1，表示原始大小

        # ROS initialization
        rospy.init_node('Image_node')
        rospy.loginfo("-------subscriber_node start!-------")
        rospy.Subscriber("/gmsl_camera/dev/video0/compressed", CompressedImage, self.msgCallback)

        self.bridge = CvBridge()

    def Unidistortion_mapping(self,calib, img_size):
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(calib.camera_matrix, calib.dist_coeffs, None,
                                                             calib.camera_matrix, tuple(img_size), cv2.CV_16SC2)

    def write_txt(self,u,v):
        output_path = self.save_dir
        make_dir(output_path)
        txt_name = self.save_file_name
        with open(output_path + txt_name, 'a') as f:
            f.write(str(u) + ',' + str(v) + '\n')

    def mouse_callback(self, event, u, v, flags, param):
        self.g_u = u
        self.g_v = v
        if event == cv2.EVENT_LBUTTONDOWN:
            print("u :", u, ", v :", v)
            self.temp_coords.append((u, v))  # 将坐标添加到临时列表中

    def msgCallback(self, msg):
        if not self.flag:
            ### Compressed image
            np_arr = np.fromstring(msg.data, np.uint8)
            img_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # ### Raw image
            # img_msg = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_size = self.img_size[::-1]
            weight_height = list(img_msg.shape[:2])
            print(img_size, weight_height)
            if img_size != weight_height:
                img_msg = cv2.resize(img_msg,img_size)

            self.img_msg = cv2.undistort(img_msg, self.calib.camera_matrix, self.calib.dist_coeffs, None, self.calib.camera_matrix)
            # undistorted_img = cv2.remap(img_msg, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)
            # self.img_msg = cv2.line(img_msg, (int(weight_height[1]/2), 0), (int(weight_height[1]/2), int(weight_height[0])), (255,0,0), thickness=1)
            
            self.flag = True

    def img_process(self, cur_img_msg):
        cv2.putText(cur_img_msg,"("+str(self.g_u)+","+str(self.g_v)+")",(self.g_u, self.g_v),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),4,lineType=cv2.LINE_AA)
        
        y_offset = cur_img_msg.shape[0] - 30  # 设置 y 偏移量，使文本显示在图像的底部
        for idx, coord in enumerate(self.temp_coords):
            text = "({},{})".format(coord[0], coord[1])
            cv2.putText(cur_img_msg, text, (10 + idx*150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, lineType=cv2.LINE_AA)

        window_name = "cur_img_msg"
        ##full screen
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        ## window screen
        # cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        cv2.imshow(window_name, cur_img_msg)
        
        k = cv2.waitKey(1)
        if k == ord('r'):  # 检查是否按下 'r' 键
            for coord in self.temp_coords:
                self.write_txt(coord[0], coord[1])
            self.temp_coords.clear()  # 清空临时坐标列表
        elif k == ord('e'):
            if self.temp_coords:
                self.temp_coords.pop()

        elif k == 27:
            exit()
        
    def main(self):
        while not rospy.is_shutdown():
            if self.flag:
                cur_img_msg = self.img_msg
                self.img_process(cur_img_msg)
                self.flag =False

if __name__ == "__main__":
    calib_dir = './camera_param/IONIQ5/f60.txt'
    save_dir = f'./output/{FOLDER_NAME}/'
    ##### 0 means original size other should [x, y]
    img_size = [1920,1080]
    save_file_name = f'img_points.txt'
    image_click = Image_click(calib_dir, save_dir, img_size, save_file_name)
    image_click.main()