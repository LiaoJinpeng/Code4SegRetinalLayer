import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize, to_categorical
from sklearn.model_selection import train_test_split

from Networks.network import UNet
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
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

# %% Build Neural Network and Train.
model = UNet(image_shape=(size_x, size_y, 1), out_cls=n_class)()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat, batch_size=1, epochs=10,
                    validation_data=(x_valid, y_valid_cat), shuffle=False)

# %% Evaluation and Output results for demonstration.
input_str = x_valid[0]
test_label = model(np.expand_dims(input_str, axis=0))
test_label_argmax = np.argmax(test_label[0], axis=-1)
val_loss, val_acc = model.evaluate(x_valid, y_valid_cat, batch_size=1)
print("Accuracy is: {}%".format(val_acc * 100.0))  # Accuracy is =  95.11 %

plt.figure()
plt.imshow(test_label_argmax)
plt.title('pred-mask')
plt.show()

plt.figure()
plt.imshow(np.argmax(y_valid_cat[0], axis=-1))
plt.title('true-mask')
plt.show()

plt.figure()
plt.imshow(input_str, 'gray')
plt.title('input-str')
plt.show()

color_maps = [
    (0, 0, 0),      # Black
    (1, 0, 0),      # Red
    (0, 1, 0),      # Lime
    (0, 0, 1),      # Blue
    (1, 1, 0),      # Yellow
    (0.5, 0, 0.5),  # Purple
    (0, 0.5, 0.5),  # Teal
    (1, 0.5, 0),    # Orange
    (0.5, 1, 0),    # Light Green
    (0.5, 0, 1),    # Violet
]

input_str = input_str[:, :, 0]
cmap = mcolors.ListedColormap(color_maps)
plt.imshow(input_str, 'gray')
plt.imshow(test_label_argmax, cmap=cmap, alpha=0.4)
plt.title('Overlapped Mask on Input Structural')
plt.savefig('Overlapped Mask on Input Structural.png')
plt.show()
