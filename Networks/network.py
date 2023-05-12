# -*- coding: utf-8 -*-
""" Objective: 
Created on (2022/08/02 - 11:58:04)

@author: JINPENG LIAO
E-mail: jyliao@dundee.ac.uk

Script Descriptions: 

"""

# %% System Setup
import os
import sys
import tensorflow as tf
import numpy as np
# import Layers
import Networks.Layers as Layers
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


# TODO: PSPNet, DeepLabV3, 
class UNet:
    """
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
    """
    def __init__(self, image_shape, out_cls):
        self.image_shape = image_shape
        self.out_cls = out_cls

    def __call__(self, include_top=True):
        Inputs = tf.keras.Input(self.image_shape)
        n_filters = [32, 32, 64, 64, 128]

        x = Layers.ConvBlock(use_BN=False)(Inputs)

        # 512 * 512 *64
        x = Layers.ConvBlock(fs=n_filters[0])(x)
        x = Layers.ConvBlock(fs=n_filters[0])(x)
        concat_1 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 256 * 256 * 128
        x = Layers.ConvBlock(fs=n_filters[1])(x)
        x = Layers.ConvBlock(fs=n_filters[1])(x)
        concat_2 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 128 * 128 * 256
        x = Layers.ConvBlock(fs=n_filters[2])(x)
        x = Layers.ConvBlock(fs=n_filters[2])(x)
        concat_3 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 64 * 64 * 512
        x = Layers.ConvBlock(fs=n_filters[3])(x)
        x = Layers.ConvBlock(fs=n_filters[3])(x)
        concat_4 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 32 * 32 * 1024
        x = Layers.ConvBlock(fs=n_filters[4])(x)
        x = Layers.ConvBlock(fs=n_filters[4])(x)

        x = Layers.TransposeConv(fs=n_filters[3])(x)  # 64 * 64 * 512
        x = tf.keras.layers.Concatenate()([x, concat_4])
        x = Layers.ConvBlock(fs=n_filters[3])(x)
        x = Layers.ConvBlock(fs=n_filters[3])(x)

        x = Layers.TransposeConv(fs=n_filters[2])(x)  # 128 * 128 * 256
        x = tf.keras.layers.Concatenate()([x, concat_3])
        x = Layers.ConvBlock(fs=n_filters[2])(x)
        x = Layers.ConvBlock(fs=n_filters[2])(x)

        x = Layers.TransposeConv(fs=n_filters[1])(x)  # 256 * 256 *128
        x = tf.keras.layers.Concatenate()([x, concat_2])
        x = Layers.ConvBlock(fs=n_filters[1])(x)
        x = Layers.ConvBlock(fs=n_filters[1])(x)

        x = Layers.TransposeConv(fs=n_filters[0])(x)  # 512 * 512 * 64
        x = tf.keras.layers.Concatenate()([x, concat_1])
        x = Layers.ConvBlock(fs=n_filters[0])(x)
        x = Layers.ConvBlock(fs=n_filters[0])(x)

        x = Layers.Conv(fs=self.out_cls, ks=1, s=1)(x)

        if include_top:
            x = Layers.get_activation_layer("softmax")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x, name="UNet")
        return models


class TransUNet:
    """https://arxiv.org/abs/2102.04306"""
    def __init__(self, image_shape, out_cls, num_heads=8):
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.out_cls = out_cls

    def __call__(self, include_top=True):
        # Hit: ver2 use the [64, 128, 256, 512, 1024] for better performance.
        n_filters = [128, 256, 512, 1024]
        hidden_size = n_filters[-1]
        Inputs = tf.keras.Input(self.image_shape)

        x1 = Layers.ConvBlock(fs=n_filters[0], ks=7, s=2)(Inputs)  # 256*256*128
        x2 = Layers.ConvBlock(fs=n_filters[1], ks=3, s=2)(x1)  # 128*128*256
        x3 = Layers.ConvBlock(fs=n_filters[2], ks=3, s=2)(x2)  # 64*64*512
        x4 = Layers.ConvBlock(fs=n_filters[3], ks=3, s=2)(x3)  # 32*32*1024

        # Vision Transformer Block
        x = tf.reshape(tensor=x4, shape=(
            tf.shape(x4)[0], tf.shape(x4)[1] * tf.shape(x4)[2], tf.shape(x4)[3]
        ))
        drop_list = [x for x in np.linspace(0, 0.1, int(len(n_filters)))]
        for i in range(int(len(n_filters))):
            x = Layers.CCT_TransformerBlock(
                num_heads=self.num_heads, hidden_size=hidden_size,
                mlp_dims=hidden_size, factor=2, drop_prob=drop_list[i]
            )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.reshape(tensor=x, shape=(
            tf.shape(x4)[0], tf.shape(x4)[1], tf.shape(x4)[2], tf.shape(x4)[3]
        ))

        x = tf.keras.layers.Concatenate()([x, x4])
        x = Layers.TransposeConv(fs=n_filters[2])(x)  # 64, 64, 512

        x = tf.keras.layers.Concatenate()([x, x3])
        x = Layers.TransposeConv(fs=n_filters[1])(x)  # 128, 128, 256

        x = tf.keras.layers.Concatenate()([x, x2])
        x = Layers.TransposeConv(fs=n_filters[0])(x)  # 256, 256, 128

        x = tf.keras.layers.Concatenate()([x, x1])
        x = Layers.TransposeConv(fs=64)(x)
        x = Layers.ConvBlock(fs=64)(x)

        x = Layers.Conv(fs=self.out_cls)(x)
        if include_top:
            if self.out_cls == 1:
                x = Layers.get_activation_layer("sigmoid")(x)
            elif self.out_cls > 1:
                x = Layers.get_activation_layer("softmax")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x,
                                       name="TransUNet")
        return models


class SwinUNet:
    """ Image Restoration Using Swin Transformer
    https://homes.esat.kuleuven.be/~konijn/publications/2021/Liang5.pdf
    """

    def __init__(self, image_shape, out_cls=None,
                 patch_size=4, num_heads=4, window_size=8):
        width, height, channel = image_shape[0], image_shape[1], image_shape[2]
        self.image_shape = (width, height, channel)
        self.out_cls = out_cls

        self.patch_size = patch_size
        # self.num_heads = num_heads
        self.drop_prob = 0.03

        self.feature_num = [96, 96 * 2, 96 * 4, 96 * 8]
        self.head_nums = [3, 6, 12, 24]
        self.window_size = self._auto_window_size(window_size)

        x_min_shape = width // np.power(2, int(len(self.feature_num) - 1))
        x_min_shape = x_min_shape // 4
        if x_min_shape < self.window_size:
            self.min_win_size = x_min_shape
        else:
            self.min_win_size = self.window_size

    def _auto_window_size(self, input_win_size):
        im_size = self.image_shape[0]
        future_size = [im_size / 4, im_size / 8, im_size / 16]
        devide_zero = np.array([x % input_win_size for x in future_size])
        if np.sum(devide_zero) > 0:
            suitable_paras = []
            ready_list = np.linspace(1, input_win_size, input_win_size)
            for num in range(len(ready_list)):
                test_zero = np.array([x % ready_list[num] for x in future_size])
                if np.sum(test_zero) > 0:
                    continue
                else:
                    suitable_paras.append(ready_list[num])
            return int(np.max(np.array(suitable_paras)))
        else:  # in this situation, the input win size is correct for network.
            return input_win_size

    def swin_block(self, x, features, num_patch_x, num_patch_y, num_heads,
                   window_size, block_num, block):
        shift_size = window_size // 2
        stage = 0
        for i in range(block_num):
            x = Layers.ShiftWindowsTransformerBlock(
                hidden_size=features,
                num_patches=(num_patch_y, num_patch_x),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                drop_rate=self.drop_prob,
                block=block,
                stage=stage,
            )(x)
            stage += 1

            x = Layers.ShiftWindowsTransformerBlock(
                hidden_size=features,
                num_patches=(num_patch_y, num_patch_x),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                drop_rate=self.drop_prob,
                block=block,
                stage=stage,
            )(x)
            stage += 1

        return x

    def __call__(self, include_top=True):
        num_patches = (self.image_shape[0] // self.patch_size) ** 2
        filter_size = (self.patch_size ** 2) * self.image_shape[-1]
        feature_num = self.feature_num  # [96, 96*2, 96*4, 96*8]
        heads = self.head_nums  # [3, 6, 12, 24]
        block = 0
        stage = 0

        # Shallow Feature Extraction
        Inputs = tf.keras.Input(self.image_shape)  # [256, 256, 1]
        patches = Layers.ConvPatchExtractor(
            patch_size=self.patch_size,
            hidden_size=filter_size,
            num_patches=num_patches)(Inputs)  # [64, 64, 16]
        projection = Layers.Dense(units=feature_num[0])(patches)
        x = tf.keras.layers.LayerNormalization()(projection)  # [64 * 64, 64]
        BS = tf.shape(x)[0]

        num_patch_y = (self.image_shape[1] // self.patch_size)  # 64
        num_patch_x = (self.image_shape[0] // self.patch_size)  # 64
        x = self.swin_block(x, feature_num[0], num_patch_x, num_patch_y,
                            heads[0], self.window_size, block_num=2, block=0)
        x_skip_1 = x

        x = Layers.PatchMerging((num_patch_x, num_patch_y), feature_num[0],
                                block=0, stage=0)(x)
        num_patch_y = num_patch_y // 2  # 32
        num_patch_x = num_patch_x // 2  # 32
        x = self.swin_block(x, feature_num[1], num_patch_x, num_patch_y,
                            heads[1], self.window_size, block_num=2, block=1)
        x_skip_2 = x

        x = Layers.PatchMerging((num_patch_x, num_patch_y), feature_num[1],
                                block=1, stage=1)(x)
        num_patch_y = num_patch_y // 2  # 16
        num_patch_x = num_patch_x // 2  # 16
        x = self.swin_block(x, feature_num[2], num_patch_x, num_patch_y,
                            heads[2], self.window_size, block_num=2, block=2)
        x_skip_3 = x

        x = Layers.PatchMerging((num_patch_x, num_patch_y), feature_num[2],
                                block=2, stage=2)(x)
        num_patch_y = num_patch_y // 2  # 8, bottleneck area
        num_patch_x = num_patch_x // 2  # 8, bottleneck area
        x = self.swin_block(x, feature_num[3], num_patch_x, num_patch_y,
                            heads[3], self.min_win_size, block_num=2, block=3)

        x = Layers.DenseExpandLayer2X(units=self.feature_num[2] * 4)(x)
        num_patch_y = num_patch_y * 2  # 16
        num_patch_x = num_patch_x * 2  # 16
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[2]])
        x = self.swin_block(x, feature_num[2], num_patch_x, num_patch_y,
                            heads[2], self.window_size, block_num=2, block=4)
        x = x + x_skip_3

        x = Layers.DenseExpandLayer2X(units=self.feature_num[1] * 4)(x)
        num_patch_y = num_patch_y * 2  # 32
        num_patch_x = num_patch_x * 2  # 32
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[1]])
        x = self.swin_block(x, feature_num[1], num_patch_x, num_patch_y,
                            heads[1], self.window_size, block_num=2, block=5)
        x = x + x_skip_2

        x = Layers.DenseExpandLayer2X(units=self.feature_num[0] * 4)(x)
        num_patch_y = num_patch_y * 2  # 64
        num_patch_x = num_patch_x * 2  # 64
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[0]])
        x = self.swin_block(x, feature_num[0], num_patch_x, num_patch_y,
                            heads[0], self.window_size, block_num=2, block=6)
        x = x + x_skip_1

        x = tf.reshape(x, shape=[BS, num_patch_x, num_patch_y, feature_num[0]])
        x = Layers.DenseExpandLayer4X(units=self.feature_num[0] * 16)(x)
        num_patch_y = num_patch_y * self.patch_size  # 256
        num_patch_x = num_patch_x * self.patch_size  # 256
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[0]])

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(self.out_cls)(x)
        x = tf.reshape(tensor=x, shape=(
            tf.shape(x)[0], num_patch_x, num_patch_y, self.out_cls))

        if include_top:
            if self.out_cls == 1:
                x = Layers.get_activation_layer('sigmoid')(x)
            elif self.out_cls > 1:
                x = Layers.get_activation_layer('softmax')(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x,
                                       name='SwinUNet')
        return models


class LightweightUShapeSwinTransformer:
    """https://ieeexplore.ieee.org/abstract/document/9999672"""
    def __init__(self, image_shape, out_cls=None, patch_szie=4, window_size=8):

        width, height, channel = image_shape[0], image_shape[1], image_shape[2]
        self.image_shape = (width, height, channel)
        self.out_cls = out_cls

        self.patch_size = patch_szie
        self.drop_prob = 0.03

        self.feature_num = [64, 64 * 2, 64 * 4, 64 * 8]
        self.head_nums = [2, 4, 8, 16]
        self.window_size = self._auto_window_size(window_size)

        x_min_shape = width // np.power(2, int(len(self.feature_num) - 1))
        x_min_shape = x_min_shape // 4
        if x_min_shape < self.window_size:
            self.min_win_size = x_min_shape
        else:
            self.min_win_size = self.window_size

    def _auto_window_size(self, input_win_size):
        im_size = self.image_shape[0]
        future_size = [im_size / 4, im_size / 8, im_size / 16]
        devide_zero = np.array([x % input_win_size for x in future_size])
        if np.sum(devide_zero) > 0:
            suitable_paras = []
            ready_list = np.linspace(1, input_win_size, input_win_size)
            for num in range(len(ready_list)):
                test_zero = np.array([x % ready_list[num] for x in future_size])
                if np.sum(test_zero) > 0:
                    continue
                else:
                    suitable_paras.append(ready_list[num])
            return int(np.max(np.array(suitable_paras)))
        else:  # in this situation, the input win size is correct for network.
            return input_win_size

    def swin_block(self, x, features, num_patch_x, num_patch_y, num_heads,
                   window_size, block_num, block):
        shift_size = window_size // 2
        stage = 0
        for i in range(block_num):
            x = Layers.ShiftWindowsTransformerBlock(
                hidden_size=features,
                num_patches=(num_patch_y, num_patch_x),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                drop_rate=self.drop_prob,
                block=block,
                stage=stage,
            )(x)
            stage += 1

            x = Layers.ShiftWindowsTransformerBlock(
                hidden_size=features,
                num_patches=(num_patch_y, num_patch_x),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                drop_rate=self.drop_prob,
                block=block,
                stage=stage,
            )(x)
            stage += 1

        return x

    def __call__(self, include_top=True):
        num_patches = (self.image_shape[0] // self.patch_size) ** 2
        filter_size = (self.patch_size ** 2) * self.image_shape[-1]
        feature_num = self.feature_num
        heads = self.head_nums
        block = 0
        stage = 0

        # Shallow Feature Extraction
        Inputs = tf.keras.Input(self.image_shape)  # [256, 256, 1]
        patches = Layers.ConvPatchExtractor(
            patch_size=self.patch_size,
            hidden_size=filter_size,
            num_patches=num_patches)(Inputs)  # [64, 64, 16]
        projection = Layers.Dense(units=feature_num[0])(patches)
        x = tf.keras.layers.LayerNormalization()(projection)  # [64 * 64, 64]
        BS = tf.shape(x)[0]

        num_patch_y = (self.image_shape[1] // self.patch_size)  # 64
        num_patch_x = (self.image_shape[0] // self.patch_size)  # 64
        x = self.swin_block(x, feature_num[0], num_patch_x, num_patch_y,
                            heads[0], self.window_size, block_num=1, block=0)
        x_skip_1 = x

        x = Layers.PatchMerging((num_patch_x, num_patch_y), feature_num[0],
                                block=0, stage=0)(x)
        num_patch_y = num_patch_y // 2  # 32
        num_patch_x = num_patch_x // 2  # 32
        x = self.swin_block(x, feature_num[1], num_patch_x, num_patch_y,
                            heads[1], self.window_size, block_num=1, block=1)
        x_skip_2 = x

        x = Layers.PatchMerging((num_patch_x, num_patch_y), feature_num[1],
                                block=1, stage=1)(x)
        num_patch_y = num_patch_y // 2  # 16
        num_patch_x = num_patch_x // 2  # 16
        x = self.swin_block(x, feature_num[2], num_patch_x, num_patch_y,
                            heads[2], self.window_size, block_num=1, block=2)
        x_skip_3 = x

        x = Layers.PatchMerging((num_patch_x, num_patch_y), feature_num[2],
                                block=2, stage=2)(x)
        num_patch_y = num_patch_y // 2  # 8, bottleneck area
        num_patch_x = num_patch_x // 2  # 8, bottleneck area
        x = self.swin_block(x, feature_num[3], num_patch_x, num_patch_y,
                            heads[3], self.min_win_size, block_num=1, block=3)

        x = Layers.DenseExpandLayer2X(units=self.feature_num[2] * 4)(x)
        num_patch_y = num_patch_y * 2  # 16
        num_patch_x = num_patch_x * 2  # 16
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[2]])
        x = self.swin_block(x, feature_num[2], num_patch_x, num_patch_y,
                            heads[2], self.window_size, block_num=1, block=4)
        x = x + x_skip_3

        x = Layers.DenseExpandLayer2X(units=self.feature_num[1] * 4)(x)
        num_patch_y = num_patch_y * 2  # 32
        num_patch_x = num_patch_x * 2  # 32
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[1]])
        x = self.swin_block(x, feature_num[1], num_patch_x, num_patch_y,
                            heads[1], self.window_size, block_num=1, block=5)
        x = x + x_skip_2

        x = Layers.DenseExpandLayer2X(units=self.feature_num[0] * 4)(x)
        num_patch_y = num_patch_y * 2  # 64
        num_patch_x = num_patch_x * 2  # 64
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[0]])
        x = self.swin_block(x, feature_num[0], num_patch_x, num_patch_y,
                            heads[0], self.window_size, block_num=1, block=6)
        x = x + x_skip_1

        x = tf.reshape(x, shape=[BS, num_patch_x, num_patch_y, feature_num[0]])
        x = Layers.DenseExpandLayer4X(units=self.feature_num[0] * 16)(x)
        num_patch_y = num_patch_y * self.patch_size  # 256
        num_patch_x = num_patch_x * self.patch_size  # 256
        x = tf.reshape(x, shape=[BS, num_patch_x * num_patch_y, feature_num[0]])

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(self.out_cls)(x)
        x = tf.reshape(tensor=x, shape=(
            tf.shape(x)[0], num_patch_x, num_patch_y, self.out_cls))

        if include_top:
            if self.out_cls == 1:
                x = Layers.get_activation_layer("sigmoid")(x)
            elif self.out_cls > 1:
                x = Layers.get_activation_layer("softmax")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x,
                                       name='LUSwinTransformer')
        return models
