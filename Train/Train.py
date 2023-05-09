# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:12:50 2022

@author: JINPENG LIAO

-	Functions:
1.	Define the loss function for network training (Input: Validation&Output)
2.	Define the Optimizer of network training
3.	Define the Network Training Type: 
    a) Supervised 
    b) Semi-supervised , Currently not support
    c) Unsupervised
4.  Define the Network Architecture (Include Return Networks)

"""
# %% System Setup
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


# %%  AIO Unsupervised Training Pipeline
class UnsupervisedTrain(tf.keras.Model):
    def __init__(self, discriminator, generator, g_loss_weight=1e-3):
        super(UnsupervisedTrain, self).__init__()
        self.D = discriminator
        self.G = generator
        self.factor = (1 / g_loss_weight)

    def compile(self, d_optimizer, g_optimizer,
                d_loss_function, g_loss_function, s_loss_function,
                metrics_fn=None):
        super(UnsupervisedTrain, self).compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.D_loss = d_loss_function
        self.G_loss = g_loss_function
        self.S_loss = s_loss_function

        self.metrics_fn = metrics_fn

    @tf.function
    def test_step(self, datasets):
        valid_input, valid_true = datasets
        valid_pred = self.G(valid_input, training=False)

        for i in range(len(self.metrics_fn)):
            self.metrics_fn[i].update_state(valid_true, valid_pred)

        return {m.name: m.result() for m in self.metrics_fn}

    @tf.function
    def train_step(self, datasets):
        train_input, train_true = datasets

        with tf.GradientTape(persistent=True) as tape:
            train_pred = self.G(train_input, training=True)

            real_logit = self.D(train_true, training=True)
            fake_logit = self.D(train_pred, training=True)

            d_loss = self.D_loss(real_out=real_logit, fake_out=fake_logit)
            g_loss = self.G_loss(real_out=real_logit, fake_out=fake_logit)

            s_loss = self.S_loss(train_pred, train_true)
            t_loss = s_loss + g_loss

        d_variables = self.D.trainable_variables
        g_variables = self.G.trainable_variables

        d_grads = tape.gradient(d_loss, d_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, d_variables))

        g_grads = tape.gradient(t_loss, g_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, g_variables))

        return {
            "DLoss": d_loss,
            "GLoss": g_loss * self.factor,
            "SLoss": s_loss,
        }


# %% AIO Supervised Training Pipeline
class SupervisedTrain(tf.keras.Model):
    def __init__(self, model=None):
        super(SupervisedTrain, self).__init__()
        self.model = model

    def compile(self, optimizers, loss_function, metrics_fn=None):
        super(SupervisedTrain, self).compile()
        self.optimizer = optimizers
        self.loss_fn = loss_function
        self.metrics_fn = metrics_fn

    @tf.function
    def test_step(self, datasets):
        for i in range(len(self.metrics_fn)):
            self.metrics_fn[i].reset_state()

        valid_input, valid_true = datasets
        valid_pred = self.model(valid_input, training=False)

        for i in range(len(self.metrics_fn)):
            self.metrics_fn[i].update_state(valid_true, valid_pred)

        return {m.name: m.result() for m in self.metrics_fn}

    @tf.function
    def train_step(self, datasets):
        train_input, train_label = datasets

        with tf.GradientTape() as tape:
            train_pred = self.model(train_input, training=True)
            loss = self.loss_fn(train_pred, train_label)

        variables = self.model.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return {"Loss:": loss}

