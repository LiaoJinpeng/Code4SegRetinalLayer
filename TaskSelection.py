# -*- coding: utf-8 -*-
""" Objective: 
Created on (yy/mm/dd - H:M:S)

@author: JINPENG LIAO
E-mail: jyliao@dundee.ac.uk

Script Descriptions: 

"""
import sys
import tensorflow as tf
import Configuration.Configs as Configs
import Configuration.DataLoader as Loader
import Train.Losses as Loss
import Networks.network as Net

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


class ReturnTrainedParas:
    def __init__(self, include_top=False, data_fp=None):
        assert isinstance(include_top, bool)
        self.include_top = include_top
        self.use_logit = bool(1 - int(include_top == True))

        self.v = Configs.Variables()
        self.image_shape = (self.v.width, self.v.height, self.v.channel)
        self.seg_num = self.v.seg_num  # number of class in segmentation
        self.data_fp = data_fp

        self.epoch_sv = self.v.fitParas["epoch_sv"]  # sv: supervised
        self.batch_size = self.v.fitParas["bs"]

        print("Used Dataset For Neural Network Train: \n")
        for key in data_fp:
            print("{}: {}".format(key, data_fp[key]))

    # ======================================================================== #
    def get_adam_optimizer(self):
        return Loss.ExtractOptimizers(
            learning_rate=self.v.optimParas['learn_rate'],
            learn_schedule=None,  # v.lr_schedule
            beta1=self.v.optimParas['beta1'],
            beta2=self.v.optimParas['beta2'],
        ).return_adam()

    def get_adamw_optimizer(self):
        return Loss.ExtractOptimizers(
            learning_rate=self.v.optimParas['learn_rate'],
            weight_decay=self.v.adamW_weight_decay,
        ).return_adamw()

    # ======================================================================== #
    def get_loss_function(self, loss_id=0):
        """
        Selection of loss_id:

            [0]    CrossEntropy,  # good to use

            [1]    SparseCrossEntropy,

            [2]    DiceCoefficientLoss,

            [3]    FocalLoss,

            [4]  DiceBCELoss,

        """

        loss_list = [
            Loss.CrossEntropy(from_logit=self.use_logit),
            Loss.SparseCrossEntropy(from_logit=self.use_logit),

            Loss.DiceCoefficientLoss(),
            Loss.FocalLoss(from_logit=self.use_logit),
            Loss.DiceBCELoss(from_logit=self.use_logit),
        ]

        return loss_list[loss_id]

    # ======================================================================== #
    def get_data_loader(self):
        return Loader.DataInputPipelineVTest()

    # ======================================================================== #
    def get_network(self, net_id=0):
        """

            [0]  UNet

            [1]  TransUNet

            [2]  SwinUNet

            [3]  LUSwinTransformer

        """
        do_not_delete_this_line = self.epoch_sv
        network_list = [
            Net.UNet,
            Net.TransUNet,
            Net.SwinUNet,
            Net.LightweightUShapeSwinTransformer,
        ]

        return network_list[net_id]
