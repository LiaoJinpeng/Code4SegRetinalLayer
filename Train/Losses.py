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


# %% Optimizers
class ExtractOptimizers:
    def __init__(self, learning_rate=1e-4, learn_schedule=None,
                 beta1=0.9, beta2=0.999, weight_decay=1e-4):
        self.learning_rate = learning_rate
        self.learn_schedule = learn_schedule
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

    def return_adam(self):
        if self.learn_schedule is not None:
            return tf.keras.optimizers.Adam(
                learning_rate=self.learn_schedule,
                beta_1=self.beta1,
                beta_2=self.beta2,
            )
        else:
            return tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta1,
                beta_2=self.beta2,
            )

    def return_adamw(self):
        return tfa.optimizers.AdamW(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
        )


# %% Loss function classes
class MSELoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=None, reduce="auto"):
        super(MSELoss, self).__init__(reduction=reduce, name="L2Loss")
        self.control_weight = control_weight
        self.loss_fn = tf.keras.losses.mse

    def call(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        loss = self.control_weight * tf.reduce_mean(loss)
        return loss


class MAELoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=None, reduce="auto"):
        super(MAELoss, self).__init__(reduction=reduce, name="L1Loss")
        self.control_weight = control_weight
        self.loss_fn = tf.keras.losses.mae

    def call(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        loss = self.control_weight * tf.reduce_mean(loss)
        return loss


class SmoothMAELoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=None, delta_=1.0,
                 reduce="auto"):
        super(SmoothMAELoss, self).__init__(reduction=reduce, name="SmoothL1")
        self.control_weight = control_weight
        self.delta_ = delta_

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        difference = y_true - y_pred
        abs_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            condition=tf.less(abs_difference, self.delta_),
            x=0.5 * squared_difference,
            y=abs_difference - 0.5,
        )
        return tf.reduce_mean(loss, axis=-1) * self.control_weight


class SSIMLoss(tf.losses.Loss):
    def __init__(self, control_weight=1.0, from_logit=None, max_val=1.,
                 reduce="auto"):
        super(SSIMLoss, self).__init__(reduction=reduce, name="SSIMLoss")
        self.control_weight = control_weight
        self.max_val = max_val
        self.loss_fn = tf.image.ssim

    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(self.loss_fn(
            img1=y_true,
            img2=y_pred,
            max_val=self.max_val
        ))
        loss = (1.0 - loss) * self.control_weight
        return loss


class SobelKernelLoss(tf.losses.Loss):
    def __init__(self, control_weight=0.1, from_logit=None, reduce='auto'):
        super(SobelKernelLoss, self).__init__(reduction=reduce,
                                              name="SobelLoss")
        self.control_weight = control_weight
        self.sobel_kernel = tf.constant([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]],
                                        dtype=tf.float32)
        self.kernel = tf.expand_dims(self.sobel_kernel, axis=-1)
        self.kernel = tf.expand_dims(self.kernel, axis=-1)

        self.pixel_loss = MSELoss(control_weight=1., reduce='none')

    def call(self, y_true, y_pred):
        pixel_loss = self.pixel_loss(y_pred, y_true)
        sobel_loss = self.pixel_loss(
            tf.nn.conv2d(y_true, self.kernel, strides=[1, 1], padding='VALID'),
            tf.nn.conv2d(y_pred, self.kernel, strides=[1, 1], padding='VALID'),
        )
        loss = pixel_loss + self.control_weight * sobel_loss
        return loss


class VGGContentLoss(tf.losses.Loss):
    def __init__(self, input_shape, control_weight=0.01, from_logit=None,
                 get_layer="block5_conv2", reduce="auto"):
        super(VGGContentLoss, self).__init__(reduction=reduce, name="VGGLoss")
        self.input_shape = (input_shape[0], input_shape[1], 3)
        self.control_weight = control_weight
        self.get_layer = get_layer
        self.backbone = self.call_backbone()

    def call_backbone(self):
        backbone = tf.keras.applications.VGG19(
            weights="imagenet", include_top=False, input_shape=self.input_shape)
        # Disable the training
        backbone.trainable = False
        for layer in backbone.layers:
            layer.trainable = False

        model = tf.keras.models.Model(
            inputs=backbone.input, outputs=backbone.get_layer(
                self.get_layer).output)
        return model

    def call(self, y_true, y_pred):
        y_true = tf.keras.layers.Concatenate(axis=-1)([y_true, y_true, y_true])
        y_pred = tf.keras.layers.Concatenate(axis=-1)([y_pred, y_pred, y_pred])
        feature_real = self.backbone(y_true)
        feature_fake = self.backbone(y_pred)
        loss = tf.reduce_mean(tf.keras.losses.mse(feature_real, feature_fake))
        loss = loss * self.control_weight
        return loss


class IRSupervisedLoss(tf.losses.Loss):
    def __init__(self, input_shape, alpha=1.0, beta=0.01, flags='mae',
                 from_logit=None, get_layer="block5_conv2"):
        super(IRSupervisedLoss, self).__init__(reduction="auto", name="IRLoss")
        self.vgg_loss_function = VGGContentLoss(input_shape=input_shape,
                                                control_weight=beta,
                                                get_layer=get_layer,
                                                reduce="none")
        if flags == 'mse':
            self.pixel_loss_function = MSELoss(control_weight=alpha,
                                               reduce="none")
        elif flags == 'mae':
            self.pixel_loss_function = MAELoss(control_weight=alpha,
                                               reduce="none")
        elif flags == 'ssim':
            self.pixel_loss_function = SSIMLoss(control_weight=alpha,
                                                reduce="none")

    def call(self, y_true, y_pred):
        vgg_loss = self.vgg_loss_function(y_true, y_pred)
        pixel_loss = self.pixel_loss_function(y_true, y_pred)
        loss = vgg_loss + pixel_loss
        return loss


class PixelLoss(tf.losses.Loss):
    def __init__(self, alpha=1., beta=0.1, from_logit=None, reduce='auto'):
        super(PixelLoss, self).__init__(reduction=reduce, name="PixelLoss")
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = MSELoss(control_weight=alpha, reduce='none')
        self.ssim_loss = SSIMLoss(control_weight=beta, reduce='none')

    def call(self, y_true, y_pred):
        ssim_loss = self.ssim_loss(y_true, y_pred)
        mse_loss = self.mse_loss(y_true, y_pred)
        loss = mse_loss + ssim_loss
        return loss


class RelativisticGANLosses():
    def __init__(self, control_weight=0.001, epsilon=1e-6):
        self.control_weight = control_weight
        self.epsilon = epsilon
        print(
            "User-Warning from Jinpeng: If you are using RasGAN loss, ensure "
            "your discriminator does not include the final-activation-layer, "
            "like:" 'sigmoid or', 'softmax')

    def discriminator_loss(self, real_out, fake_out):
        real_average_out = K.mean(real_out, axis=0)
        fake_average_out = K.mean(fake_out, axis=0)

        Real_Fake_relativistic_average_out = real_out - fake_average_out
        Fake_Real_relativistic_average_out = fake_out - real_average_out

        real_loss = K.mean(K.log(K.sigmoid(Real_Fake_relativistic_average_out) +
                                 self.epsilon), axis=0)
        fake_loss = K.mean(
            K.log(1 - K.sigmoid(Fake_Real_relativistic_average_out) +
                  self.epsilon), axis=0)  # Use 0.9 to replace 1.0

        D_loss = -(real_loss + fake_loss)
        return D_loss

    def generator_loss(self, real_out, fake_out):
        real_average_out = K.mean(real_out, axis=0)
        fake_average_out = K.mean(fake_out, axis=0)

        Real_Fake_relativistic_average_out = real_out - fake_average_out
        Fake_Real_relativistic_average_out = fake_out - real_average_out

        real_loss = K.mean(K.log(K.sigmoid(Fake_Real_relativistic_average_out) +
                                 self.epsilon), axis=0)
        fake_loss = K.mean(
            K.log(1 - K.sigmoid(Real_Fake_relativistic_average_out) +
                  self.epsilon), axis=0)

        G_loss = -(real_loss + fake_loss)
        G_loss = G_loss * self.control_weight
        return G_loss


class GANLosses():
    def __init__(self, control_weight=0.001, epsilon=1e-6, from_logit=False):
        self.control_weight = control_weight
        self.epsilon = epsilon
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logit)

    def discriminator_loss(self, real_out, fake_out):
        real_loss = self.bce(tf.ones_like(real_out), real_out)
        fake_loss = self.bce(tf.zeros_like(fake_out), fake_out)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, real_out, fake_out):
        fake_loss = self.bce(tf.ones_like(fake_out), fake_out)
        fake_loss = fake_loss * self.control_weight
        return fake_loss


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
