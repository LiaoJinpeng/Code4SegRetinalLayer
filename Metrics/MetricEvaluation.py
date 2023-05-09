import tensorflow as tf
from Configuration.Configs import Variables
from Networks.network import UNet
from TaskSelection import ReturnTrainedParas as Paras

v = Variables()
paras = Paras(include_top=True, data_fp=v.project_mice_wound_segmentation)

model = UNet(paras.image_shape, paras.seg_num)(paras.include_top)
model.load_weights('UNet')

