# -*- coding: utf-8 -*-
""" Objective: Neural Layers and Neural Blocks for Network Architecture - v2

Created on (2022/08/01 - 15:06)

@author: JINPENG LIAO
E-mail: jyliao@dundee.ac.uk

Script Descriptions:
This script provide the self-define neural layers and neural blocks, for the
network architecture design. Compared with the v1 'utils', the script aims to
initialize the neural layer based on the 'tf.keras.layers.Layer' class

Notes:
- Dilated Convolution: convolution with holes
    Can achieve the larger receptive file while not increase the parameters.
    Work well for the semantic segmentation task.
- Depth-wise Separable Convolution: combine the depth-wise and point-wise
    Can be used to extract the feature, while the paramters and calculation cost
    are reduced and lower. Always used on lightweight network (e.g. MobileNet)
    For depth-wise convolution, each kernel will only focus on the single
    channel, and point-wise will create a 1x1xM filters for calculation.
"""

# %% System Setup
import os
import sys
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
from keras.applications import imagenet_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


def get_activation_layer(activation_type="relu"):
    flags = activation_type.lower()

    activation_library = {
        "relu": tf.keras.layers.ReLU(),
        "leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.2),
        "prelu": tf.keras.layers.PReLU(shared_axes=[1, 2]),
        "softmax": tf.keras.layers.Softmax(),
        "sigmoid": tf.keras.activations.sigmoid,
        "tanh": tf.keras.activations.tanh,
        "gelu": tf.keras.activations.gelu,
    }

    if flags in activation_type:
        return activation_library[flags]
    else:
        raise ValueError("@Jinpeng: ERROR INPUT OF THE ACTIVATION LAYER TYPE")


# %% Convolution Layer
@tf.keras.utils.register_keras_serializable()
class Conv(tf.keras.layers.Layer):
    """ Self-define 2D Convolution Neural Layer with kernel initializer """

    def __init__(self, fs=64, ks=3, s=1, use_bias=False, padding="same",
                 **kwargs):
        """
        Args:
            fs: int, filter size of the Conv2D layer
            ks: int, kernel size of the Conv2D layer
            s: int, strides of the Conv2D layer
            use_bias: bool, decide use the bias or not inside the layer
            padding: string, optional setting: "same" or "valid"
        """
        super(Conv, self).__init__(**kwargs)
        self.he_initializer = tf.keras.initializers.HeNormal()
        self.random_initializer = tf.random_normal_initializer(0., 0.02)
        self.fs = fs
        self.ks = ks
        self.s = s
        self.use_bias = use_bias
        self.padding = padding

        self.conv2d_layer = tf.keras.layers.Conv2D(
            filters=fs, kernel_size=(ks, ks), strides=(s, s), padding=padding,
            use_bias=use_bias, kernel_initializer=self.random_initializer,
        )

    def call(self, inputs):
        x = self.conv2d_layer(inputs)
        return x

    def get_config(self):
        config = super(Conv, self).get_config()
        config.update(
            {
                "fs": self.fs,
                "ks": self.ks,
                "s": self.s,
                "use_bias": self.use_bias,
                "padding": self.padding,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Transpose Convolution Layer
@tf.keras.utils.register_keras_serializable()
class TransposeConv(tf.keras.layers.Layer):
    """ Self-define 2D Transpose Convolution Layer with kernel initializer  """

    def __init__(self, fs=64, ks=3, s=2, use_bias=False, padding="same",
                 **kwargs):
        """
        Args:
            fs: int, filter size of the Transpose Conv2D layer
            ks: int, kernel size of the Transpose Conv2D layer
            s: int, strides of the Transpose Conv2D layer
            use_bias: bool, decide use the bias or not inside the layer
            padding: string, optional setting: "same" or "valid"
        """
        super(TransposeConv, self).__init__(**kwargs)
        self.he_initializer = tf.keras.initializers.HeNormal()
        self.random_initializer = tf.random_normal_initializer(0., 0.02)
        self.fs = fs
        self.ks = ks
        self.s = s
        self.use_bias = use_bias
        self.padding = padding

        self.transpose_conv2d_layer = tf.keras.layers.Conv2DTranspose(
            filters=fs, kernel_size=(ks, ks), strides=(s, s), padding=padding,
            use_bias=use_bias, kernel_initializer=self.random_initializer,
        )

    def call(self, inputs):
        x = self.transpose_conv2d_layer(inputs)
        return x

    def get_config(self):
        config = super(TransposeConv, self).get_config()
        config.update(
            {
                "fs": self.fs,
                "ks": self.ks,
                "s": self.s,
                "use_bias": self.use_bias,
                "padding": self.padding,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Dense Layer
@tf.keras.utils.register_keras_serializable()
class Dense(tf.keras.layers.Layer):
    """ Customn define dense layer with initializer """

    def __init__(self, units=256, use_bias=False, **kwargs):
        """
        Args:
            units: int, the unit in the dense layer.
            use_bias: bool, decide use the bias or not.
        """
        super(Dense, self).__init__(**kwargs)
        self.he_initializer = tf.keras.initializers.HeNormal()
        self.random_initializer = tf.random_normal_initializer(0., 0.02)
        self.units = units
        self.use_bias = use_bias

        self.dense_layer = tf.keras.layers.Dense(
            units=units, use_bias=use_bias,
            kernel_initializer=self.random_initializer
        )

    def call(self, inputs):
        x = self.dense_layer(inputs)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class PatchMerging(tf.keras.layers.Layer):
    """ Patch Merging Layer is used after each SwinTransformer Block Stage.
    When finish one Swin-Transformer stage, will do the down-sample operation.
    For example, the input shape is (56, 56, channels), the down-sample is to do
    1). extract the left-up, right-up, left-bottom, right-bottom features and
    merge together.
    2). processing by a layer-normalization, and dense-linear layer.
    The processing can be seen as the weighted-pooling layer.
    """

    def __init__(self, num_patch, embed_dim, block="0", stage="0"):
        super(PatchMerging, self).__init__(
            name="PatchMeringLayer" +
                 "block_" + str(block) + "_stage_" + str(stage))
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = Dense(units=2 * embed_dim, use_bias=False)
        self.layer_normalization = layers.LayerNormalization()

    def call(self, x):
        height, width = self.num_patch
        _, _, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, c))

        # from start-to-end of the array, collect the feature as:
        # start:end:step. In below function, the end is not set,
        # hence, it will be started from 0/1 with step=2.
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C

        x = tf.concat((x0, x1, x2, x3), axis=-1)  # B, H/2, W/2, 4*C
        x = self.layer_normalization(x)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * c))
        x = self.linear_trans(x)  # B, H/2 * W/2, 4*C

        return x

    def get_config(self):
        config = super(PatchMerging, self).get_config()
        config.update({
            "num_patch": self.num_patch,
            "embed_dim": self.embed_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% ViT Trainable Patch Extractor
@tf.keras.utils.register_keras_serializable()
class ConvPatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, num_patches, **kwargs):
        super(ConvPatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = num_patches

        self.extract_conv2d_layer = Conv(
            fs=self.hidden_size, ks=self.patch_size, s=self.patch_size,
            use_bias=False, padding="valid",
        )

    def call(self, inputs):
        patches = self.extract_conv2d_layer(inputs)
        patches = tf.reshape(tensor=patches,
                             shape=(-1, self.num_patches, self.hidden_size))
        return patches

    def get_config(self):
        config = super(ConvPatchExtractor, self).get_config()
        config.update({
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "num_patches": self.num_patches,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% MLP Block
@tf.keras.utils.register_keras_serializable()
class MLPBlock(tf.keras.layers.Layer):
    """ Multilayer perceptron Blocks for feed-forward network """

    def __init__(self, factor=None, units=256, droprate=0.3, acti="gelu",
                 **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        self.units = units
        self.droprate = droprate
        self.factor = factor

        self.dense_layer_1 = Dense(units=int(self.units * self.factor))
        self.dense_layer_2 = Dense(units=self.units)

    def call(self, x):
        # Default use the tf.nn.gelu as activation function, more function
        # block to change the activation layer will be finish in the future
        if self.factor is not None:
            x = self.dense_layer_1(x)
            x = tf.nn.gelu(x)
            x = tf.keras.layers.Dropout(self.droprate)(x)

            x = self.dense_layer_2(x)
            x = tf.keras.layers.Dropout(self.droprate)(x)

            return x

    def get_config(self):
        config = super(MLPBlock, self).get_config()
        config.update({
            "units": self.units,
            "droprate": self.droprate,
            "factor": self.factor,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Attention Blocks
@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    """ Self-attention head blocks """

    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def call(self, Q, K, V):
        score = tf.matmul(Q, K, transpose_b=True)
        dims = tf.cast(tf.shape(K)[-1], dtype=score.dtype)
        score = score / tf.math.sqrt(dims)

        probs = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(probs, V)
        return output

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi-head attention """

    def __init__(self, num_heads, hidden_size, droprate=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.droprate = droprate

        self.query_dense_layer = Dense(units=self.hidden_size)
        self.key_dense_layer = Dense(units=self.hidden_size)
        self.value_dense_layer = Dense(units=self.hidden_size)
        self.proj_dense_layer = Dense(units=self.hidden_size)

    def _separate_head(self, inputs, batch_size):
        # Example for what this function do:
        # Input shape: (Batch Size, 256, 786)
        # Reshape ==>> (Batch Size, 256, 12, 786//12)
        assert self.hidden_size % self.num_heads == 0

        proj_dims = self.hidden_size // self.num_heads
        x = tf.reshape(tensor=inputs,
                       shape=(batch_size, -1, self.num_heads, proj_dims))
        # Transpose => (Batch Size, 12, 256, 786//12)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        Q = self.query_dense_layer(inputs)
        K = self.key_dense_layer(inputs)
        V = self.value_dense_layer(inputs)

        Q = self._separate_head(inputs=Q, batch_size=batch_size)
        K = self._separate_head(inputs=K, batch_size=batch_size)
        V = self._separate_head(inputs=V, batch_size=batch_size)

        score = SelfAttention()(Q, K, V)
        score = tf.transpose(score, perm=[0, 2, 1, 3])
        score = tf.reshape(score, (batch_size, -1, self.hidden_size))

        output = self.proj_dense_layer(score)
        if self.droprate is not None:
            output = tf.keras.layers.Dropout(self.droprate)(output)

        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "droprate": self.droprate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Transformer Blocks
@tf.keras.utils.register_keras_serializable()
class CCT_TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, hidden_size, mlp_dims, factor=2, droprate=0.3,
                 drop_prob=0.05, **kwargs):
        super(CCT_TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_dims = mlp_dims
        self.factor = factor
        self.drop_prob = drop_prob

        self.layer_normalization_1 = layers.LayerNormalization()
        self.layer_normalization_2 = layers.LayerNormalization()

        self.stochastidepth_layer_1 = StochasticDepth(drop_prob=drop_prob)
        self.stochastidepth_layer_2 = StochasticDepth(drop_prob=drop_prob)

        self.mhsa_layer = MultiHeadAttention(num_heads=num_heads,
                                             hidden_size=hidden_size,
                                             droprate=droprate)
        self.mlp_block = MLPBlock(factor=factor, units=mlp_dims,
                                  droprate=droprate)

        self.adding_layer_1 = layers.Add()
        self.adding_layer_1 = layers.Add()

    def call(self, inputs):
        LN1 = self.layer_normalization_1(inputs)
        x = self.mhsa_layer(LN1)
        x = self.stochastidepth_layer_1(x)
        x = self.adding_layer_1([x, LN1])

        y = self.layer_normalization_2(x)
        y = self.mlp_block(y)
        y = self.stochastidepth_layer_2(x)
        y = self.adding_layer_1([y, x])

        return y

    def get_config(self):
        config = super(CCT_TransformerBlock, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "mlp_dims": self.mlp_dims,
            "factor": self.factor,
            "drop_prob": self.drop_prob,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Dropout for very deep neural networks
@tf.keras.utils.register_keras_serializable()
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs):
        # Define the Remain Ratio of Neurals
        keep_prob = 1 - self.drop_prob
        # Define the shape of Random Dropout: = [Batch Size, 1, 1, 1]
        shape = (tf.shape(inputs)[0],) + (1,) * (tf.shape(inputs).shape[0] - 1)
        # Create a random tensor to decide random dropout neurals
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        random_tensor = tf.floor(random_tensor)
        # Apply Random Dropout Out by Random Activatie/Inactivatie layer
        return (inputs / keep_prob) * random_tensor

    def get_config(self):
        config = super(StochasticDepth, self).get_config()
        config.update({
            "drop_prob": self.drop_prob,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Convolution Neural Network - used Blocks
@tf.keras.utils.register_keras_serializable()
class ConvBlock(tf.keras.layers.Layer):
    """ Very basic conv+bn+activation blocks, build this block for convenience
    """

    def __init__(self, fs=64, ks=3, s=1, use_bias=False, use_BN=True,
                 acti="relu", **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.fs = fs
        self.ks = ks
        self.s = s
        self.use_bias = use_bias
        self.use_BN = use_BN

        self.conv2d_layer = Conv(fs=fs, ks=ks, s=s, use_bias=use_bias)
        self.bn_layer = layers.BatchNormalization()
        self.activation = get_activation_layer(activation_type=acti)

    def call(self, inputs):
        x = self.conv2d_layer(inputs)
        if self.use_BN:
            x = self.bn_layer(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            "fs": self.fs,
            "ks": self.ks,
            "s": self.s,
            "use_bias": self.use_bias,
            "use_BN": self.use_BN,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def window_partition(x, window_size=2):
    """ Window partition function for the Shift-window Vision Transformer.

    Split the input tensor from: [height, width, channels] to ==>>
    [height/window_size, window_size, width/window_size, window_size, channels]

    Then, transpose the order of the tensor, to ==>>
    [height/window_size, width/window_size, window_size, window_size, channels]

    Finally, reshape the tensor to ==>>
    [height/window_size*width/window_size, window_size, window_size, channels]

    """
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(tensor=x, shape=(
        -1, patch_num_y, window_size, patch_num_x, window_size, channels))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    """ Window reverse function for the Shift-window Vision Transformer.

    Convert the tensor from previous divided windows to normal shape, from
    [height/window_size*width/window_size, window_size, window_size, channels]
    to: => [height, width, channels]

    """
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(tensor=windows, shape=(
        -1, patch_num_y, patch_num_x, window_size, window_size, channels))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


@tf.keras.utils.register_keras_serializable()
class WindowSelfAttention(tf.keras.layers.Layer):
    """ Window MultiHead Self Attention Layer.
    For example, the input is from the example tensor in WinMHSABlocks. And the
    shape of the input tensor is: (Batch Size*(64/4)*(64/4), 4*4, 96)
    a). use the dense layer with hidden_size*3 units to generate the qkv tensor.
    The output is called qkv with shape=(Batch Size*(64/4)*(64/4), 4*4, 96*3)
    b). assume that the num_heads is set as 3, and the qkv tensor from (a) is
    reshaped to -->> (3, Batch Size*(64/4)*(64/4), 3, 4*4, 32)
    c). then, divide the qkv tensor to q, k, v tensor, respectively. And hence
    each q, k, v tensor has shape: (Batch Size*(64/4)*(64/4), 3, 4*4, 32). And
    represent that each q/k/v tensor has 4096 windows, and each windows with the
    3 different heads, 4*4 patches and 32 dimensions feature.

    d). layer normalization for the tensor -> q
    e). then, q @ k as the first-step of the attention layer, and the shape
    of the q @ k outputs is: (Batch Size*(64/4)*(64/4), 3, 4*4, 4*4)

    """

    # TODO: Fully understand window-based self-attention
    def __init__(self, hidden_size, window_size, num_heads=6, drop_prob=0.03,
                 drop_rate=0.3, use_bias=True, **kwargs):
        super(WindowSelfAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.scale = (hidden_size // num_heads) ** -0.5

        self.qkv_dense_layer = Dense(units=hidden_size * 3, use_bias=use_bias)
        self.dropout_layer = layers.Dropout(drop_rate)
        self.projection_dense_layer = Dense(hidden_size)

    def build(self, input_shape):
        """ Obtain the relative position bias information.
        For example, the window size is set as 3 in this introduction.
        a). generate the 'coords' with shape 2 * 3 * 3. It means that in the
        window with size 3 * 3, each position has its coordinates [x, y],
        and the relative_coords with shape 2 * 9 * 9, it means in the 9 points,
        each points [x, y] and other points have the error. For example,
        [0][3][1] is represent the error difference between No.3 points and
        No.1 points. Then reshape, and let those two coordinates add the
        value in 3-1=2. That is because the range of those coords is [0, 2].
        Then, let the coords-y multiply the (2*3-1=5). Finally, add the error
        difference value in x,y dimensions, and obtain the
        relative_position_index, 3^2 * 3^2, are the relative position bias
        between two points. finally, find the position in bias_table.
        b). add the relative_position to the attention output, and apply the
        softmax on it.
        c). ((q @ k) + relative_position)/sqrt(dims) @ v. Then the output shape
        is: (Batch Size*(64/4)*(64/4), 3, 4*4, 96)
        """
        num_window_elements = (2 * self.window_size[0] - 1) * (
                2 * self.window_size[1] - 1)  # (2*2-1) * (2*2-1) = 9
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
            name="name_for_run" +
                 str(np.random.rand(1000)[np.random.randint(1000)]),
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None,
                                                       :]
        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # shape from (2, 4, 4) ==>> (4, 4, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(  # (4, 4) matrix
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name="name_for_run" +
                 str(np.random.rand(1000)[np.random.randint(1000)])
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dims = channels // self.num_heads
        qkv = self.qkv_dense_layer(x)
        qkv = tf.reshape(qkv, shape=(-1, size, 3, self.num_heads, head_dims))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements,
                                           num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias,
                                              perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            # mask shape: (1024, 4, 1)
            # after expand_dims: (1, 1024, 1, 4, 1)
            mW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1),
                                                axis=0), dtype=tf.float32)
            # attn after reshape: (BS*, 1024, 4, 64, 64)
            attn = (tf.reshape(attn,
                               shape=(-1, mW, self.num_heads, size, size)) +
                    mask_float)
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = tf.keras.activations.softmax(attn, axis=-1)
        else:
            attn = tf.keras.activations.softmax(attn, axis=-1)

        attn = self.dropout_layer(attn)

        attn_output = attn @ v
        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, shape=(-1, size, channels))
        attn_output = self.projection_dense_layer(attn_output)
        attn_output = self.dropout_layer(attn_output)

        return attn_output

    def get_config(self):
        config = super(WindowSelfAttention, self).get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "drop_prob": self.drop_prob,
            "drop_rate": self.drop_rate,
            "use_bias": self.use_bias,
            "scale": self.scale,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ShiftWindowsTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_patches, num_heads, window_size=7,
                 shift_size=0, mlp_units=1024, use_bias=True, drop_rate=0.0,
                 block="0", stage="0", **kwargs):
        """ When the shift_size == 0, this is a window-MSA block:
        For example, the input shape is (Batch size, 64*64, 96), H=W=64
        a). the input will be first processed by layer-normalization.
        b). then, reshape to the (Batch Size, 64, 64, 96).
        c). use the defined 'function' -- 'window_partition' to divide the
        tensor from (Batch Size, 64, 64, 96) to ==>> ("assume window size = 4")
        (Batch Size*(64/4)*(64/4), 4, 4, 96)
        d). reshape the output from (c) -->> (Batch Size*(64/4)*(64/4), 4*4, 96)
        e). finally, use the WindowAttention Layer to process the tensor. For
        the shift_size==0, the attn_mask is set as 'None'.

        f). ....

        """
        super(ShiftWindowsTransformerBlock, self).__init__(
            name=("SwinTransformerBasicLayer" +
                  "block_" + str(block) + "_stage_" + str(stage)), **kwargs)
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.layer_normalization_1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowSelfAttention(
            hidden_size=hidden_size, window_size=(window_size, window_size),
            num_heads=num_heads, use_bias=use_bias, drop_rate=drop_rate,
        )
        self.drop_path = StochasticDepth(drop_rate)
        self.layer_normalization_2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp_block = MLPBlock(factor=4, units=hidden_size)

        if min(self.num_patches) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patches)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            """ Create the attn_mask
            assume the input shape is (8^2*B, 7^2, 96), the input tensor for the
            attention layer. also, assume the num_heads is set as 3. Hence, the 
            input paras of the attention layer is 'x_windows' and 'attn_mask'.
            
            To generate the attn_mask:
            a) the shape of the mask should be (1, 56, 56, 1), which is the same
            as the input image shape (i.e., width and height of feature maps). 
            b) based on mask, generate the h_slices and w_slices, which with 
            value (0, -7), (-7, -3), (-3, None), after the recycle processing, 
            the mask will be divided into 9 patches and be marked. 
            
            Example of Mask for better understanding: 
            12 x 12 patches, window have 3 x 3 patches, shift size = 1, 4 x 4 windows.
            
            
            """
            height, width = self.num_patches
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1

            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(tensor=mask_windows, shape=[
                -1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=-1) - tf.expand_dims(
                mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            # shape: (height/window_size*width/window_size, window_size^2, 1)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                trainable=False,
                name="name_for_run" +
                     str(np.random.rand(1000)[np.random.randint(1000)])
            )

    def call(self, x):
        height, width = self.num_patches
        _, num_patches_before, channels = x.shape
        x_skip = x

        x = self.layer_normalization_1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            """ Reason of using the tf.roll when shift_size > 0.
            The disadvantage of the Window-MHSA is that the information from 
            the different divided patches cannot share with other patches. 
            Hence, the shift window is that shifting the feature map but not 
            the windows (i.e. the different patches in the same windows). 
            
            One of the methods is to adding the zero padding into the feature 
            maps and do the feature map rolling. However, it will increase 
            the computing cost. Hence, proposed by the paper, use the rolling 
            method to shift the feature map to-left and to-upper to move half 
            of the window size. (e.g., move 3 patch if window size is 7). 
            """
            shifted_x = tf.roll(input=x,
                                shift=[-self.shift_size, -self.shift_size],
                                axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size *
                                                 self.window_size, channels))

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, shape=(
            -1, self.window_size, self.window_size, channels))

        shifted_x = window_reverse(attn_windows, self.window_size, height,
                                   width, channels)

        if self.shift_size > 0:
            x = tf.roll(input=shifted_x,
                        shift=[self.shift_size, self.shift_size],
                        axis=[1, 2])
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x + x_skip
        x_skip = x

        x = self.layer_normalization_2(x)
        x = self.mlp_block(x)
        x = self.drop_path(x)
        x = x + x_skip

        return x

    def get_config(self):
        config = super(ShiftWindowsTransformerBlock, self).get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_patches": self.num_patches,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "shift_size": self.shift_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DenseExpandLayer4X(tf.keras.layers.Layer):
    def __init__(self, units, bias=False, upscale=4, **kwargs):
        super(DenseExpandLayer4X, self).__init__(**kwargs)
        self.units = units
        self.bias = bias
        self.upscale = upscale

        self.layer_normalization = layers.LayerNormalization()
        self.expand_layer = layers.Dense(units=self.units, use_bias=self.bias)

    def call(self, inputs):
        B1 = tf.shape(inputs)[0]
        C1 = tf.shape(inputs)[-1]
        C2 = self.units
        ishape1 = tf.shape(inputs)[1]
        US = self.upscale

        x = tf.reshape(tensor=inputs, shape=[B1, ishape1 * ishape1, C1])
        x = self.expand_layer(x)  # BS, ishape*ishape, C1 * 16

        x = tf.reshape(tensor=x,
                       shape=[B1, ishape1, ishape1, US, US, C2 // (US * US)])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(tensor=x,
                       shape=[B1, ishape1 * US, ishape1 * US, C2 // (US * US)])
        x = tf.reshape(tensor=x, shape=[B1, -1, C2 // (US * US)])

        x = self.layer_normalization(x)
        x = tf.reshape(tensor=x,
                       shape=[B1, ishape1 * US, ishape1 * US, C2 // (US * US)])
        return x

    def get_config(self):
        config = super(DenseExpandLayer4X, self).get_config()
        config.update({
            "units": self.units,
            "bias": self.bias,
            "upscale": self.upscale,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DenseExpandLayer2X(tf.keras.layers.Layer):
    def __init__(self, units, bias=False, upscale=2, **kwargs):
        super(DenseExpandLayer2X, self).__init__(**kwargs)
        self.units = units
        self.bias = bias
        self.upscale = upscale

        self.layer_normalization = layers.LayerNormalization()
        self.expand_layer = layers.Dense(units=self.units, use_bias=self.bias)

    def call(self, inputs):
        B1 = tf.shape(inputs)[0]
        C1 = tf.shape(inputs)[-1]
        C2 = self.units
        ishape1 = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
        ishape1 = tf.cast(tf.math.sqrt(ishape1), dtype=tf.int32)
        US = self.upscale

        x = tf.reshape(tensor=inputs, shape=[B1, ishape1 * ishape1, C1])
        x = self.expand_layer(x)  # BS, ishape*ishape, C1 * 4

        x = tf.reshape(tensor=x,
                       shape=[B1, ishape1, ishape1, US, US, C2 // (US * US)])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(tensor=x,
                       shape=[B1, ishape1 * US, ishape1 * US, C2 // (US * US)])
        x = tf.reshape(tensor=x, shape=[B1, -1, C2 // (US * US)])

        x = self.layer_normalization(x)
        x = tf.reshape(tensor=x,
                       shape=[B1, ishape1 * US, ishape1 * US, C2 // (US * US)])
        return x

    def get_config(self):
        config = super(DenseExpandLayer2X, self).get_config()
        config.update({
            "units": self.units,
            "bias": self.bias,
            "upscale": self.upscale,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)
