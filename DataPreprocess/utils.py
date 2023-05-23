import re
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy.io import loadmat


def read_str_data(fp='D'):
    s = fp.split('/')[-1]  # obtain the filename
    p = re.search('\((\d+)mmx(\d+)mm\)', s, re.IGNORECASE)

    w = p.group(1)
    h = p.group(2)

    img_shape = None
    if w == h and w == '3':
        img_shape = (300, 1536, 300)
    elif w == h and (w == '6' or w == '9' or w == '12'):
        img_shape = (500, 1536, 500)
    elif w == '15' and h == '9':
        img_shape = (834, 1536, 500)

    str_img = np.fromfile(fp, np.uint8)
    str_img = np.reshape(str_img, img_shape)

    str_img = np.swapaxes(str_img, 0, 1)
    str_img = np.swapaxes(str_img, 1, 2)
    str_img = np.flip(str_img, axis=0)
    str_img = np.flip(str_img, axis=1)

    return str_img  # Output shape is [nZ, nY, nX]


def read_seg_layer(fp='D'):
    file_format = fp.split('.')[-1]
    layers = None

    if file_format == 'npy':
        layers = np.load(fp)
    elif file_format == 'mat':
        layers = loadmat(fp)['layers']

    return layers


def output_color_mask(ori_count, z_s, str_img=None, seg_img=None, save_fp=None):
    # TODO: Change z_s, z_e if position is incorrect.
    z_s = z_s
    z_e = z_s + 500

    str_img = str_img[z_s:z_e, :, :]
    frame_num = np.shape(str_img)[1]

    count = ori_count
    template = "{}\\Frame_{}.png"
    for frame in range(frame_num):
        current_str = str_img[:, frame, :]  # Save it
        mask_shape = np.shape(current_str)
        mask = np.zeros(mask_shape, dtype=np.uint8)  # Save it

        layer_e = np.array(seg_img[frame, :, :]) - z_s
        layer_s = np.concatenate([
            np.zeros((frame_num, 1)), layer_e[:, :-1]], axis=-1)

        tmp = 0
        for i in range(0, 9):
            for x_p in range(frame_num):
                s_p = int(np.ceil(layer_s[x_p, i]))
                e_p = int(np.ceil(layer_e[x_p, i]))
                mask[s_p: e_p, x_p] = tmp
            tmp += 1
        for x_p in range(frame_num):
            e_p = int(np.ceil(layer_e[x_p, i]))
            mask[e_p:, x_p] = tmp

        io.imsave(fname=(save_fp + template.format('valid', str(count))),
                  arr=mask)
        io.imsave(fname=(save_fp + template.format('input', str(count))),
                  arr=current_str)
        count += 1


basic_fp = "F:\\OneDrive - UW\\SegmentationTask\\ManualSeg\\"
process_id = 4
str_fps = [
    "P1273_Angio (6mmx6mm)_6-13-2019_13-2-0_OS_sn25281_cube_z.img",
    "P2457_Angio (6mmx6mm)_8-26-2021_12-58-39_OS_sn61468_cube_z.img",
    "P2457_Angio (6mmx6mm)_8-26-2021_13-9-17_OD_sn61481_cube_z.img",
    "P3275_Angio (6mmx6mm)_6-27-2019_10-47-51_OS_sn26417_cube_z.img",
    "P3352_Angio (6mmx6mm)_2-10-2022_12-12-47_OS_sn66347_cube_z.img",
]
seg_fps = [
    "P1273_Angio (6mmx6mm)_6-13-2019_13-2-0_OS_sn25281_cube_z\\layers.npy",
    "P2457_Angio (6mmx6mm)_8-26-2021_12-58-39_OS_sn61468_cube_z\\layers.npy",
    "P2457_Angio (6mmx6mm)_8-26-2021_13-9-17_OD_sn61481_cube_z\\layers.npy",
    "P3275_Angio (6mmx6mm)_6-27-2019_10-47-51_OS_sn26417_cube_z\\layers.npy",
    "P3352_Angio (6mmx6mm)_2-10-2022_12-12-47_OS_sn66347_cube_z\\layers.npy",
]
counts = [x * 502 for x in range(len(seg_fps))]
crop_start = [
    700, 450, 300, 370, 400
]

str_img = read_str_data(basic_fp + str_fps[process_id])
seg_img = read_seg_layer(basic_fp + seg_fps[process_id])

seg_img = np.delete(seg_img, 6, axis=-1)
seg_img = np.delete(seg_img, 6, axis=-1)

new_seg_img = np.zeros(np.shape(seg_img))
new_seg_img[:, :, 0:6] = seg_img[:, :, 0:6]
new_seg_img[:, :, 6] = seg_img[:, :, 8]
new_seg_img[:, :, 7:9] = seg_img[:, :, 6:8]

output_color_mask(
    ori_count=counts[process_id], z_s=crop_start[process_id],
    str_img=str_img, seg_img=new_seg_img,
    save_fp="F:\\OneDrive - UW\\SegmentationTask\\gitCode4SegRetinalLayer\\Cod"
            "e4SegRetinalLayer\\Dataset\\retina seg test data\\")
