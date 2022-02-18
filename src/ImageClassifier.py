#!/usr/bin/env python

import cv2
import csv
import numpy as np
import shutil
import os
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int8


class ClassifyImage:
    def __init__(self):
        rospy.init_node("ClassifyImg", anonymous=True)
        train_data = []
        # TrainFolder = "~/catkin_ws/src/shivam_final_project/Launch/crop_train/"
        with open("train.txt", "r") as f:
            reader = csv.reader(f)
            lines = list(reader)
        # this line reads in all print("Img Output Class ->",1)images listed in the file in GRAYSCALE, and resizes them to 33x25 pixels
        for i in range(len(lines)):
            train = []
            I = cv2.resize(cv2.imread("./crop_train/" +
                                      lines[i][0] + ".png", 1), (33, 25))
            for j in np.arange(3):
                Im = I[:, :, j] - np.mean(I[:, :, j])
                Im = Im.flatten()
                In = Im/np.linalg.norm(Im)
                # train.append(I[:, :, j])
                train.append(In)
            train = np.array(train).reshape(33 * 25 * 3)
            train_data.append(train)
        train_data = np.array(train_data).reshape(
            len(lines), 33 * 25 * 3).astype(np.float32)

        # read in training labels
        train_labels = np.array([np.int32(lines[i][1])
                                 for i in range(len(lines))])

        # Train classifier
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        self.ImageClass = rospy.Publisher(
            "/ImageClass", Int8, queue_size=1)
        self.img_sub = rospy.Subscriber(
            "/raspicam_node/image/compressed", CompressedImage, self.imgCallback, queue_size=1)

    def imgCallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        c_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_read = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        img_width, img_height = np.array(img_read.shape)
        img = cv2.addWeighted(img_read, 2.5, np.zeros(
            img_read.shape, img_read.dtype), 0, 0)
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_max = -1
        area_max = -1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if img_width*1/5 < w < img_width*4/5 and img_height*1/5 < h < img_height*4/5:
                if w * h > area_max:
                    if 0.80 < w / h < 1.20:
                        area_max = w * h
                        cnt_max = cnt
        # bound the images
        if area_max == -1:
            pass
        else:
            x, y, w, h = cv2.boundingRect(cnt_max)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            c_img = c_img[y:y + h, x:x + w]
        test = []
        I = cv2.resize(c_img, (33, 25))
        for j in np.arange(3):
            Im = I[:, :, j] - np.mean(I[:, :, j])
            Im = Im.flatten()
            In = Im/np.linalg.norm(Im)
            # train.append(I[:, :, j])
            test.append(In)
        test = np.array(test).reshape(1, 33 * 25 * 3)
        test_img = test.astype(np.float32)
        ret, _, _, _ = self.knn.findNearest(test_img, 3)
        print("Image classified as ", ret)
        self.ImageClass.publish(ret)


ClassifyImg = ClassifyImage()
rospy.spin()
