# -*- coding: utf-8 -*-
""" Input the Tensor Datasets from the PNG/JPG format files

Created on Tue Jun 28 12:46:59 2022

@author: JINPENG LIAO

"""

# %% System Setup
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Configuration import Configs
from keras.utils import to_categorical

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")
variables = Configs.Variables
AUTOTUNE = tf.data.experimental.AUTOTUNE


# %% Zip Datasets
class DataLoader:
    def __init__(self, image_fp, label_fp, valid_rate=0.2):
        self.image_fp = image_fp
        self.label_fp = label_fp
        self.valid_rate = valid_rate

        self.image_fps = glob(os.path.join(image_fp, '*.png'))
        self.label_fps = glob(os.path.join(label_fp, '*.png'))

        v = variables()
        self.num_of_data = v.num_of_ds
        self.imagesize = (v.width, v.height)
        self.seg_cls = v.seg_num

    def __call__(self):
        images = []
        labels = []

        "Data Loading from filepath"
        for image_fp in self.image_fps[:self.num_of_data]:
            image = cv2.imread(image_fp, 0)
            image = cv2.resize(
                image, self.imagesize, interpolation=cv2.INTER_NEAREST)
            images.append(image)

        for label_fp in self.label_fps[:self.num_of_data]:
            label = cv2.imread(label_fp, 0)
            label = cv2.resize(
                label, self.imagesize, interpolation=cv2.INTER_NEAREST)
            labels.append(label)

        images, labels = np.array(images), np.array(labels)
        print("Image & Label Loaded, Number of Segment-Class:{}".format(
            np.unique(labels)))

        "Encode Labels to One-hot"
        labelencoder = LabelEncoder()
        n, h, w = labels.shape

        labels_reshaped = labels.reshape(-1, 1)
        labels_reshaped_encoded = labelencoder.fit_transform(labels_reshaped)
        labels_encoded_original_shape = labels_reshaped_encoded.reshape(n, h, w)

        "Create Training Data and Validation Data"
        images = np.expand_dims(images, axis=-1)
        images = images / 255.
        labels = np.expand_dims(labels_encoded_original_shape, axis=3)

        # train-input, valid-input, train-label, valid-label
        x_train, x_valid, y_train, y_valid = train_test_split(
            images, labels, test_size=self.valid_rate, random_state=0)

        train_labels_cat = to_categorical(y_train, num_classes=self.seg_cls)
        y_train_cat = train_labels_cat.reshape((
            y_train.shape[0], y_train.shape[1], y_train.shape[2], self.seg_cls))
        valid_labels_cat = to_categorical(y_valid, num_classes=self.seg_cls)
        y_label_cat = valid_labels_cat.reshape((
            y_valid.shape[0], y_valid.shape[1], y_valid.shape[2], self.seg_cls))

        return x_train, y_train_cat, x_valid, y_label_cat
