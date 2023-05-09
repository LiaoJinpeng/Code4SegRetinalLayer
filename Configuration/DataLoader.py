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
import pathlib
import cv2
import json
from glob import glob
from matplotlib import pyplot as plt
# import Configs
from Configuration import Configs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")
variables = Configs.Variables
AUTOTUNE = tf.data.experimental.AUTOTUNE


# %% Zip Datasets
def zip_and_batch_ds(inputs_x, inputs_y, batch_sizes):
    ds = tf.data.Dataset.from_tensor_slices((inputs_x, inputs_y))
    return ds.batch(batch_sizes)


def func(ds):
    return next(iter(ds))


class ISDataLoader:
    def __init__(self, data_filepath):
        self.v = variables()

        self.c = self.v.channel
        self.w = self.v.width
        self.h = self.v.height
        self.nc = self.v.seg_num
        self.img_dtypes = tf.dtypes.uint8
        self.img_shape = [self.w, self.h]

        self.train_ds_num = self.v.num_of_ds["train"]
        self.valid_ds_num = self.v.num_of_ds["valid"]

        self.train_fp = data_filepath["train_data"]
        self.valid_fp = data_filepath["valid_data"]

        self.bs = self.v.fitParas['bs']

    def return_fps(self):
        train_input = sorted(glob(os.path.join(self.train_fp, "input/*"))
                             )[: self.train_ds_num]
        train_label = sorted(glob(os.path.join(self.train_fp, "valid/*"))
                             )[: self.train_ds_num]
        valid_input = sorted(glob(os.path.join(self.valid_fp, "input/*"))
                             )[: self.valid_ds_num]
        valid_label = sorted(glob(os.path.join(self.valid_fp, "valid/*"))
                             )[: self.valid_ds_num]

        train_input = tf.data.Dataset.from_tensor_slices(train_input)
        train_label = tf.data.Dataset.from_tensor_slices(train_label)
        valid_input = tf.data.Dataset.from_tensor_slices(valid_input)
        valid_label = tf.data.Dataset.from_tensor_slices(valid_label)

        return train_input, train_label, valid_input, valid_label

    def return_dataset(self):
        train_input, train_label, valid_input, valid_label = self.return_fps()
        train_input = func(train_input.map(
            self.process_image).cache().repeat().batch(self.train_ds_num))
        train_label = func(train_label.map(
            self.process_label).cache().repeat().batch(self.train_ds_num))
        valid_input = func(valid_input.map(
            self.process_image).cache().repeat().batch(self.valid_ds_num))
        valid_label = func(valid_label.map(
            self.process_label).cache().repeat().batch(self.valid_ds_num))

        train_ds = zip_and_batch_ds(train_input, train_label, self.bs)
        valid_ds = zip_and_batch_ds(valid_input, valid_label, self.bs)

        train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, valid_ds

    def process_image(self, ds):
        ds = tf.io.read_file(ds)
        ds = tf.image.decode_jpeg(ds, self.c)
        ds = tf.image.resize(ds, size=self.img_shape, antialias=True)
        ds = tf.clip_by_value(ds / 255., 0., 1.)
        return ds

    def process_label(self, ds):
        ds = tf.io.read_file(ds)
        ds = tf.image.decode_png(ds, self.nc, dtype=self.img_dtypes)
        ds = tf.image.resize(ds, size=self.img_shape, antialias=True)
        return ds


class DataLoader4Segmentation:
    def __init__(self, data_filepath):
        self.v = variables()

        self.c = self.v.channel
        self.w = self.v.width
        self.h = self.v.height
        self.seg_cls = self.v.seg_num
        self.img_dtypes = tf.dtypes.uint8
        self.img_shape = [self.w, self.h]

        self.train_ds_num = self.v.num_of_ds["train"]
        self.valid_ds_num = self.v.num_of_ds["valid"]

        self.train_fp = data_filepath["train_data"]
        self.valid_fp = data_filepath["valid_data"]

        self.bs = self.v.fitParas['bs']

    def return_fps(self):
        train_input = sorted(
            glob(os.path.join(self.train_fp, "input/*")))[: self.train_ds_num]
        train_label = sorted(
            glob(os.path.join(self.train_fp, "valid/*")))[: self.train_ds_num]
        valid_input = sorted(
            glob(os.path.join(self.valid_fp, "input/*")))[: self.valid_ds_num]
        valid_label = sorted(
            glob(os.path.join(self.valid_fp, "valid/*")))[: self.valid_ds_num]

        return train_input, train_label, valid_input, valid_label

    def read_image(self, fps):
        images = []
        for fp in fps:
            tmp_image = np.asarray(cv2.resize(
                cv2.imread(fp), (self.w, self.h)), dtype=np.float32)
            tmp_image = tmp_image / 255.
            images.append(tmp_image)
        return images

    def read_label(self, fps):
        labels = []
        for fp in fps:
            read_label = cv2.imread(fp, 0)
            one_hot_label = np.zeros((self.h, self.w, self.seg_cls))
            for i, unique_value in enumerate(np.unique(read_label)):
                one_hot_label[:, :, i][read_label == unique_value] = 1
            labels.append(one_hot_label)
        return labels

    def return_dataset(self):
        train_input, train_label, valid_input, valid_label = self.return_fps()
        train_images = np.array(self.read_image(train_input), dtype=np.float32)
        valid_images = np.array(self.read_image(valid_input), dtype=np.float32)
        train_labels = np.array(self.read_label(train_label), dtype=np.int32)
        valid_labels = np.array(self.read_label(valid_label), dtype=np.int32)

        train_ds = zip_and_batch_ds(train_images, train_labels, self.bs)
        valid_ds = zip_and_batch_ds(valid_images, valid_labels, self.bs)
        return train_ds, valid_ds


