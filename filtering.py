import shutil
import hyperspectral
import envi
import scipy.ndimage
import numpy

def main():
    #env1= "data\\davar\\a_train_b_test\\filtering\\a_mosaic"
    env1="data\\davar\\a_train_b_test\\filtering\\c_mosaic"

    mask_file= "data\\davar\\a_train_b_test\\filtering\\low_res_mask.bmp"

    env1_filt = box_filter(env1 , 5)
    size = subsample(env1_filt, mask_file,5)
    print(size)

    # env2_filt= box_filter(env2 , 5)
    # size = subsample(env2_filt, mask_file)
    # print(size)

def box_filter (envi_file , n):
    E = envi.envi(envi_file)
    D = E.loadall()
    if E.header.interleave is not 'bsq':
        print("ERROR: filtering is just implemented for BSQ!")
    # D_filter = scipy.ndimage.uniform_filter(D, size=[5, 5, 1], mode='constant', cval=0.0)
    k = numpy.ones([1, n, n], dtype='float32') * 1.0 / (float(n)*float(n))  # definne box filter
    D_filter = scipy.ndimage.filters.convolve(D, k, mode='reflect')  # convolve the hyperspectral image with box filter
    D_filter_sift = hyperspectral.sift2(D_filter)  # sift a 2D hyperspectral image into a PxB matrix
    D_filter_1D = D_filter_sift.flatten()  # flattenn the matrix to an array
    envi_file_blur = envi_file + "_blured_bsq"  # name the output
    outfile = open(envi_file_blur, 'wb')  # open a binary file to write the array
    D_filter_1D.tofile(outfile)
    outfile.close()
    # create a header file-
    shutil.copy2(envi_file + '.hdr', envi_file_blur + '.hdr')
    return envi_file_blur


def subsample(envi_file, mask_file,n):
    E = envi.envi(envi_file)
    D= E.loadall()
    Ds= D[:,::n,::n]
    Ds_sift = hyperspectral.sift2(Ds)
    Ds_flatten = Ds_sift.flatten()
    #mask = scipy.misc.imread(mask_file, flatten=True)
    #D_subsample = E.loadmask(mask).transpose()
    #D_flatten = D_subsample.flatten()
    out_name = envi_file + '_sub'
    outfile = open(out_name , 'wb')
    Ds_flatten.tofile(outfile)
    outfile.close()
    return [Ds.shape[0],Ds.shape[1], Ds.shape[2]]



if __name__ == '__main__':
    main()