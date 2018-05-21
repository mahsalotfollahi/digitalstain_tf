import matplotlib.pyplot as plt
import scipy
import scipy.misc
import numpy
import envi
import hyperspectral


def prediction_func (CLASS , envifile, maskfile= '' ):
    print("Validating Stain...")
    plt.ion()
    E = envi.envi(envifile)
    if not maskfile == "":
        E.close()  # close the ENVI file
        mask = scipy.misc.imread(maskfile, flatten=True)
        print(numpy.count_nonzero(mask))
        E = envi.envi(envifile, mask=mask)

    batch_size = 10000
    Fv = E.loadbatch(batch_size)  # load the first batch
    n = 0
    while not Fv == []:  # loop until an empty batch is returned
        if n == 0:
            Tv = CLASS.predict(Fv.transpose()).transpose()
        else:
            Tv = numpy.append(Tv, CLASS.predict(Fv.transpose()).transpose(), 1)  # append the predicted labels from this batch to those of previous batches
        COLORS = hyperspectral.unsift2(Tv, E.batchmask())  # convert the matrix of class labels to a 2D array
        RGB = numpy.rollaxis(COLORS, 0, 3).astype(numpy.ubyte)
        plt.imshow(RGB)  # display it
        plt.pause(0.05)
        Fv = E.loadbatch(batch_size)  # load the next batch
        n = n + 1
    return RGB