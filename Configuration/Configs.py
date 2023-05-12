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
import requests
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


def auto_call_phone(model):
    template = "https://api.day.app/8EHMQUNbqsxL4Z2Pzj7TZY/"
    template = template + model.name + "/_Train_Completed_"
    requests.get(template)


class Variables:
    def __init__(self):
        # TODO: parameters for keras.fit function
        self.fitParas = {
            "bs": 1,  # bs: batch size
            "epoch_sv": 20,  # sv: supervised
        }

        # TODO: number of train/valid datasets
        self.num_of_ds = {
            "train": 2,
            "valid": 2,
        }

        # TODO: shape of the datasets and labels
        self.width = 384
        self.height = 384
        self.channel = 1
        self.image_shape = (self.width, self.height, self.channel)

        # TODO: number of segmentation class
        self.seg_num = 6
        assert self.seg_num > 0

        # TODO: Optimizers Hyper-Paras
        self.optimParas = {
            "beta1": 0.9,
            "beta2": 0.999,
            "learn_rate": 1e-3,
            "decay_rate": 0.98,
            "decay_step": 40000,
        }
        # or you might prefer AdamW optimizer?
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.optimParas["learn_rate"],
            decay_steps=self.optimParas["decay_step"],
            decay_rate=self.optimParas["decay_rate"])
        self.adamW_weight_decay = 0.0001

        # TODO: Please change the filepath of prepared dataset
        self.project_mice_wound_segmentation = {
            "train_data": r"D:\Jinpeng\data_slice_massroad\train",
            "valid_data": r"D:\Jinpeng\data_slice_massroad\valid",
        }

        self.test_stage_retina_layer = {
            "train_image": r"C:\Users\24374\Desktop\retina - test\input",
            "train_label": r"C:\Users\24374\Desktop\retina - test\label",
            "valid_image": r"",
            "valid_label": r"",
        }

        # additional information.
        self.ts = time.localtime()
        self.time_date = str(self.ts[0]) + '.' + str(self.ts[1]) + '.' + str(
            self.ts[2]) + ' -- ' + str(self.ts[3]) + ':' + str(self.ts[4])

    # %% Print the Paras Setting
    def print_config(self, network=None, s_loss=None):

        template = "\n\
        Image Shape : {}*{}*{}; \n\
        Train Epoch : {}; \n\
        Batch Size  : {}; \n\
                    "
        template_hyper = "\n\
        Learn Rate : {:.6f};\n\
        Decay Rate : {:.6f};\n\
        Decay Step : {};    \n\
        beta_1 : {:.4f}; \n\
        beta_2 : {:.4f}; \n\
                          "

        print("Start Time: {}".format(self.time_date))

        print("Network Basic Variables:")
        print(template.format(
            self.width, self.height, self.channel,
            self.fitParas["epoch_sv"], self.fitParas["bs"]))

        print("Hyper Parameters Tuning: ")
        print(template_hyper.format(
            self.optimParas["learn_rate"],
            self.optimParas["decay_rate"],
            self.optimParas["decay_step"],
            self.optimParas["beta1"],
            self.optimParas["beta2"]
        ))

        if network is not None:
            print("Network Name:  {}".format(network.name))
        if s_loss is not None:
            print("Loss Function: {}".format(s_loss.name))

        saved_name = network.name + "-" + s_loss.name
        return saved_name
