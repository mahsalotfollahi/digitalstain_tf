import digitalstain_tf
import hyperspectral
import predict
import scipy
import scipy.misc , scipy.ndimage.filters
import envi
import matplotlib.pyplot as plt
import numpy


## DAVAR HIGH MAG DATA
envifile_train = "data\\davar\\a_train_b_test\\a_mosaic_base_pca60"
stainfile = "data\\davar\\a_train_b_test\\filtering\\a_tar.bmp"
envifile_test = "data\\davar\\a_train_b_test\\c_mosaic_base_bip_apca60"
dm=True
#trainmask = "data\\hsiproc-tutorial\\breast_mask.bmp"
#maskfile = "data\\davar\\secB\\mask_tissue.bmp"
#maskfile ="data\\davar\\a_train_b_test\\low_res_mask.bmp"
'''
##HSIPROC-TUTORIAL DATA.....................................................
envifile_train = "data\\hsiproc-tutorial\\crop\\breast_bip_crop_pca20"#"data\\hsiproc-tutorial\\breast_bip_pca60"
envifile_test = "data\\hsiproc-tutorial\\breast_bip_pca20"
stainfile = "data\\hsiproc-tutorial\\crop\\hne_crop.bmp" #"data\\hsiproc-tutorial\\hne.bmp"
#maskfile = "data\\hsiproc-tutorial\\tissue_mask.bmp"
#trainmask ="data\\hsiproc-tutorial\\crop\\breast_mask_half.bmp"
#maskfile=maskfile, trainmask=trainmask
'''


#downsampling
# if dm:
#     envifile_train = downsampling(envifile_train , 5)
#     envifile_test = downsampling(envifile_test , 5)
#
##..........................................................................
# C, RGB = digitalstain_tf.generate_stain(envifile, stainfile, trainmask= trainmask , maskfile = maskfile,  N=1000000, batch_size = 100000)
CLASS = digitalstain_tf.feed_forward_net(envifile_train, stainfile, N=1000000)#, trainmask=trainmask
RGB_train = predict. prediction_func (CLASS , envifile_train)
scipy.misc.imsave("data\\davar\\a_train_b_test\\train_bc60.bmp", RGB_train)
RGB_test = predict. prediction_func (CLASS , envifile_test)#, maskfile= maskfile
scipy.misc.imsave("data\\davar\\a_train_b_test\\test2_bc60.bmp", RGB_test)