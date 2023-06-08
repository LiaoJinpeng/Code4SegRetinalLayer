# -*- coding: utf-8 -*-
""" Configuration Settings for:
    - Datasets Loading
    - Networks Architecture
    - Progress Running

Created on Tue Jun 28 09:58:28 2022

@author: JINPENG LIAO

This script allows the user to set the basic Variables and Hyper-Parameters 
of the deep-learning network training.

"""
import os
import time
import sys
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


def get_early_stop(monitor='val_loss', patience=30, mode='min'):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, mode=mode, restore_best_weights=True
    )


def get_check_point(checkpoint_path, monitor='loss', mode='min'):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, monitor=monitor,
        mode=mode, save_best_only=True,
    )


class Variables:
    def __init__(self):
        # TODO: parameters for keras.fit function
        self.fitParas = {
            "bs": 4,  # bs: batch size
            "epoch_sv": 20,  # sv: supervised
        }

        # TODO: number of train/valid datasets
        self.num_of_ds = 200

        # TODO: shape of the datasets and labels
        self.width = 512
        self.height = 512
        self.channel = 1
        self.image_shape = (self.width, self.height, self.channel)

        # TODO: number of segmentation class
        self.seg_num = 10  # number of segmentation class you want. please input
        # 2 if you want to do binary segmentation/classification.

        self.optimParas = {
            "beta1": 0.9,
            "beta2": 0.999,
            "learn_rate": 1e-3,
            "decay_rate": 0.98,
            "decay_step": 40000,
        }

        # additional information.
        self.ts = time.localtime()
        self.time_date = str(self.ts[0]) + '.' + str(self.ts[1]) + '.' + str(
            self.ts[2]) + ' -- ' + str(self.ts[3]) + ':' + str(self.ts[4])

    # %% Print the Paras Setting
    def print_config(self):
        print("Start Time: {}".format(self.time_date))
        template = "\n\
        Image Shape : {}*{}*{}; \n\
        Train Epoch : {}; \n\
        Batch Size  : {}; \n\
                    "

        print("Network Basic Variables:")
        print(template.format(
            self.width, self.height, self.channel,
            self.fitParas["epoch_sv"], self.fitParas["bs"]))
