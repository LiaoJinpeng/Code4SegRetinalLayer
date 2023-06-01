# -*- coding: utf-8 -*-
""" Objective: 
Created on (yy/mm/dd - H:M:S)

@author: JINPENG LIAO
E-mail: jyliao@dundee.ac.uk

Script Descriptions: 

"""
# %% System setup
import os
import sys
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


class CrossEntropy(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=True):
        super(CrossEntropy, self).__init__(reduction="auto", name="CELoss")
        self.control_weight = control_weight
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=from_logit)

    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, tf.shape(y_true))
        loss = self.loss_fn(y_true, y_pred)
        loss = loss * self.control_weight
        return loss


class SparseCrossEntropy(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=True):
        super(SparseCrossEntropy, self).__init__(reduction="auto",
                                                 name="SpareCELoss")
        self.control_weight = control_weight
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logit)

    def call(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        loss = loss * self.control_weight
        return loss


class DiceCoefficientLoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, smooth=100):
        super(DiceCoefficientLoss, self).__init__(reduction="auto",
                                                  name="DiceLoss")
        self.control_weight = control_weight
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        intersection = K.sum(y_true * y_pred)
        dice = (2. * intersection + self.smooth) / (
                K.sum(y_true) + K.sum(y_pred) + self.smooth)

        loss = (1.0 - dice) * self.control_weight
        return loss


class FocalLoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=False):
        super(FocalLoss, self).__init__(reduction="auto", name="FocalLoss")
        self.control_weight = control_weight
        self.loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=from_logit, alpha=0.25, gamma=4.0, )

    def call(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        loss = self.control_weight * loss
        return loss


class DiceBCELoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, beta=0.5, from_logit=False):
        super(DiceBCELoss, self).__init__(reduction="auto", name="DiceBCELoss")
        self.control_weight = control_weight
        self.beta = beta

        self.dice_loss = DiceCoefficientLoss(control_weight=(1 - beta))
        self.bce_loss = K.binary_crossentropy

    def call(self, y_true, y_pred):
        dice_loss = self.dice_loss(y_true, y_pred)
        bce_loss = self.bce_loss(y_true, y_pred)
        loss = bce_loss * self.beta + dice_loss
        return loss
