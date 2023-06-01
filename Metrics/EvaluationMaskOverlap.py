import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Networks import network


size_x = 256
size_y = 256
n_class = 10

net_list = {
    'UNet': network.UNet,
    'LightweightUShapeTransformer': network.LightweightUShapeSwinTransformer,
}

saved_weight_list = {
    'UNet': 'U-Net\\UNet',
    'LightweightUShapeTransformer': 'LUST\\LUSwinTransformer',
}

maker = 'LightweightUShapeTransformer'
model = net_list[maker]((size_x, size_y, 1), n_class)()
model.load_weights(saved_weight_list[maker]).expect_partial()

fp = "E:\\jpliao.p\\Code4SegRetinalLayer\\Code4SegRetinalLayer-main\\Dataset\\"
img_fp = fp + "yuxuan_require-2.png"

img = tf.io.read_file(img_fp)
img = tf.image.decode_png(img, channels=1, dtype=tf.uint8)
img = tf.image.resize(img, [size_x, size_y])
img = img / 255.
img = tf.clip_by_value(img, 0, 1)

input_img = tf.expand_dims(img, axis=0)
pred_mask = model(input_img)

pred_mask_argmax = np.argmax(np.array(pred_mask[0]), axis=-1)

color_maps = [
    (0, 0, 0),  # Black
    (1, 0, 0),  # Red
    (0, 1, 0),  # Lime
    (0, 0, 1),  # Blue
    (1, 1, 0),  # Yellow
    (0.5, 0, 0.5),  # Purple
    (0, 0.5, 0.5),  # Teal
    (1, 0.5, 0),  # Orange
    (0.5, 1, 0),  # Light Green
    (0.5, 0, 1),  # Violet
]

input_str = img
cmap = mcolors.ListedColormap(color_maps)
plt.imshow(input_str, 'gray')
plt.imshow(pred_mask_argmax, cmap=cmap, alpha=0.4)
plt.title('Predicted Result by {}'.format(model.name))
plt.savefig('Overlapped Mask on Unseen - {}.png'.format(model.name))
plt.show()

