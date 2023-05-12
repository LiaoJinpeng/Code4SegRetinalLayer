import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize, to_categorical
from sklearn.model_selection import train_test_split

from Networks.network import UNet
from matplotlib import pyplot as plt
import sys
import random

size_x = 192
size_y = 192
n_class = 11

# %% Data Loading
fp_input = r"F:\OneDrive - UW\SegmentationTask\gitCode4SegRetinal" \
           r"Layer\Code4SegRetinalLayer\Dataset\retina seg test data\input"
fp_label = r"F:\OneDrive - UW\SegmentationTask\gitCode4SegRetinal" \
           r"Layer\Code4SegRetinalLayer\Dataset\retina seg test data\valid"

input_fps = glob.glob(os.path.join(fp_input, '*.png'))
label_fps = glob.glob(os.path.join(fp_label, '*.png'))

train_images = []
for image_fp in input_fps:
    image = cv2.imread(image_fp, 0)
    image = cv2.resize(image, (size_x, size_y))
    train_images.append(image)
train_images = np.array(train_images)

train_labels = []
for label_fp in label_fps:
    label = cv2.imread(label_fp, 0)
    label = cv2.resize(label, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
    train_labels.append(label)
train_labels = np.array(train_labels)

print('class label as:', np.unique(train_labels))

# %% Encode Labels
labelencoder = LabelEncoder()
n, h, w = train_labels.shape

train_labels_reshaped = train_labels.reshape(-1, 1)
train_labels_reshaped_encoded = labelencoder.fit_transform(
    train_labels_reshaped)
train_labels_encoded_original_shape = train_labels_reshaped_encoded.reshape(
    n, h, w)

# %% Create Dataset
train_images = np.expand_dims(train_images, axis=-1)
train_images = train_images / 255.
train_labels = np.expand_dims(train_labels_encoded_original_shape, axis=3)

x_train, x_valid, y_train, y_valid = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=0
)

train_labels_cat = to_categorical(y_train, num_classes=n_class)
y_train_cat = train_labels_cat.reshape((
    y_train.shape[0], y_train.shape[1], y_train.shape[2], n_class))

valid_labels_cat = to_categorical(y_valid, num_classes=n_class)
y_valid_cat = valid_labels_cat.reshape((
    y_valid.shape[0], y_valid.shape[1], y_valid.shape[2], n_class))

# %% Build Neural Network and Train
model = UNet(image_shape=(size_x, size_y, 1), out_cls=n_class)()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat, batch_size=1, epochs=2,
                    validation_data=(x_valid, y_valid_cat), shuffle=False)

test_label = model(x_valid[0])

# _, acc = model.evaluate(x_valid, y_valid_cat)
# print("Accuracy is = ", (acc * 100.0), "%")
