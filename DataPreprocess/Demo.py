import re
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

str_fp = "F:\\OneDrive - UW\\SegmentationTask\\ManualSeg\\P1273_Angio (" \
         "6mmx6mm)_6-13-2019_13-2-0_OS_sn25281_cube_z.img"
seg_fp = "F:\\OneDrive - UW\\SegmentationTask\\ManualSeg\\P1273_Angio (" \
         "6mmx6mm)_6-13-2019_13-2-0_OS_sn25281_cube_z\\layers.npy"

# %% Load the Structural Data/Img
# Read Img File
s = str_fp.split('/')[-1]
p = re.search('\((\d+)mmx(\d+)mm\)', s, re.IGNORECASE)  # Search the FOV of Data

# Below are manual define shape to match the data shape acquire by OCT System.
# As Yuxuan mentioned, 'img' file do not contain the shape of dataset, and hence
# we require to reshape it to -> (w, z, h) by ourself.
w = p.group(1)
h = p.group(2)
img_shape = None

if w == h and w == '3':
    img_shape = (300, 1536, 300)
elif w == h and (w =='6' or w == '9' or w == '12'):
    img_shape = (500, 1536, 500)
elif w == '15' and h == '9':
    img_shape = (834, 1536, 500)

# Based on the obtained image shape for 'img' file reshape, we can then input
# the data based on the provided filepath.
str_img = np.fromfile(str_fp, np.uint8)
str_img = np.reshape(str_img, img_shape)
# after reshape operation, the direction of img is incorrect. base on yuxuan
# provided code, I copy it and transfer str_img to corrected shape: [nz, ny, nx]
str_img = np.swapaxes(str_img, 0, 1)
str_img = np.swapaxes(str_img, 1, 2)
str_img = np.flip(str_img, axis=0)
str_img = np.flip(str_img, axis=1)

# %% Load the Seg-Layer Information, here we load the outside seg-file/info.
file_format = seg_fp.split('.')[-1]
if file_format == 'npy':
    layers = np.load(seg_fp)
    layer_num = layers.shape[-1]
elif file_format == 'mat':
    layers = loadmat(seg_fp)['layers']
    layer_num = layers.shape[-1]

# %% Now we have the Segmented Layers Information and Structural Images, we need
# to plot the layers information and structural images in one image.

z_start = 800
z_end = 1300
slice_num = 250

fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(111)

assert slice_num < str_img.shape[-1]
ax1.imshow(str_img[z_start:z_end, slice_num, :], cmap='gray')
x = np.arange(0, layers.shape[1])
cmap = matplotlib.cm.get_cmap('prism')
for i in range(0, 10):
    ax1.plot(x, (layers[slice_num, :, i] - z_start), linewidth=2,
             color=cmap(i * 10), label=str(i))

ax1.legend()
plt.show()


mask_shape = np.shape(str_img[z_start:z_end, slice_num, :])
mask = np.zeros(mask_shape, dtype=np.uint8)
new_layer_positions = layers[slice_num, :, :] - z_start
first_layer = np.zeros((500, 1))
old_layer_positions = np.concatenate(
    [first_layer, new_layer_positions[:, :-1]], axis=-1)

nums = 0
for i in range(0, 10):
    current_layer = new_layer_positions[:, i]
    for x_p in range(500):
        s_p = int(np.ceil(old_layer_positions[x_p, i]))
        e_p = int(np.ceil(new_layer_positions[x_p, i]))
        mask[s_p: e_p, x_p] = nums
    nums += 1

plt.figure()
plt.imshow(mask)
plt.show()
