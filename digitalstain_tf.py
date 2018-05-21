import hyperspectral
import envi
import classify
import numpy
import scipy
import scipy.misc, scipy.ndimage, scipy.signal

import glob
import matplotlib.pyplot as plt
import random
import tflearn
import shutil




def feed_forward_net(envifile, stainfile, trainmask="",n_epoch=50, N=5000):
    if trainmask == "":
        E = envi.envi(envifile)
    else:
        mask = scipy.misc.imread(trainmask, flatten=True)
        E = envi.envi(envifile, mask=mask)

    mask = classify.random_mask(E.mask, N)
    scipy.misc.imsave("random.bmp", mask)

    Ft = E.loadmask(mask).transpose()

    stain = numpy.rollaxis(scipy.misc.imread(stainfile), 2)
    #plt.figure()
    #plt.imshow(stain)
    Tt = hyperspectral.sift2(stain, mask).transpose()  # sift a 2D hyperspectral image into a PxB matrix where P is the number of pixels and B is the number of bands
    print("Training MLPRegressor...")
    net = tflearn.input_data(shape= [None , Ft.shape[1]])
    net = tflearn.fully_connected(net ,7 ,activation= 'relu' )#,

    net = tflearn.fully_connected(net , 3, activation= 'relu')#,
    net = tflearn.regression(net, batch_size= 128 ,  loss='mean_square', metric='R2', learning_rate=0.01)#learning_rate=0.01
    CLASS = tflearn.DNN(net, tensorboard_dir='.\log',tensorboard_verbose=2)
    CLASS.fit(Ft, Tt, n_epoch=n_epoch, show_metric=True, snapshot_step=500, validation_set=0.2)

    #if validate == False:
    return CLASS


# def downsampling (envi_file , n):
#     E = envi.envi(envi_file)
#     D = E.loadall()
#     # D_filter = scipy.ndimage.uniform_filter(D, size=[5, 5, 1], mode='constant', cval=0.0)
#     k = numpy.ones([n, n, 1], dtype='float32') * 1.0 / 25.0  # definne box filter
#     D_filter = scipy.ndimage.filters.convolve(D, k, mode='reflect')  # convolve the hyperspectral image with box filter
#     D_filter_sift = hyperspectral.sift2(D_filter)  # sift a 2D hyperspectral image into a PxB matrix
#     D_filter_1D = D_filter_sift.flatten()  # flattenn the matrix to an array
#     envi_file_blur = envi_file + "_blured_bsq"  # name the output
#     outfile = open(envi_file_blur, 'wb')  # open a binary file to write the array
#     D_filter_1D.tofile(outfile)
#     outfile.close()
#     # create a header file-
#     shutil.copy2(envi_file + '.hdr', envi_file_blur + '.hdr')
#     return envi_file_blur

