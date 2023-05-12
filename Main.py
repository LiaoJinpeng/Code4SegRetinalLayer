#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@Author: Jinpeng Liao
@Contact: jyliao@dundee.ac.uk
@Modify Time: 2/16/2023 1:44 PM
"""
import os
import sys
import tensorflow as tf

from Configuration.Configs import Variables
from TaskSelection import ReturnTrainedParas as Paras
from TaskSelection import get_early_stop

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")
v = Variables()

paras = Paras(include_top=True, data_fp=v.project_mice_wound_segmentation)
networks = paras.get_network(net_id=0)(
    paras.image_shape, paras.seg_num)(paras.include_top)
# TODO: Future work - loss function definition, based on dataloader.
loss_fun = paras.get_loss_function(loss_id=0)
# TODO: Future work - change the data loader to fit retinal data.
train_ds, valid_ds = paras.get_data_loader().return_dataset()

networks.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(False), metrics=[
        tf.keras.metrics.MeanIoU(num_classes=paras.seg_num),
    ])
networks.fit(train_ds, epochs=paras.epoch_sv, validation_data=valid_ds,
             callbacks=[get_early_stop('val_loss', 20, 'min')])

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

test_img = Image.open(r"C:\Users\24374\Desktop\retina - test\input\image1.bmp")
test_img = np.array(test_img) / 255.
test_img = test_img[450:834, 100:484]
test_img = np.expand_dims(test_img, axis=0)
test_img = np.expand_dims(test_img, axis=-1)

pred_label = networks(test_img)
pred_label = np.argmax(np.array(pred_label), axis=-1)

plt.figure()
plt.imshow(pred_label[0])
plt.show()

plt.figure()
plt.imshow(test_img[0, :, :, 0])
plt.show()
