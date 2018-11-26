import urllib
import glob
import numpy as np
from scipy import misc
from skimage import feature, img_as_float, color
from skimage.filters import roberts, sobel, scharr, prewitt, threshold_otsu, threshold_adaptive
import matplotlib.pyplot as plt
from skimage import data

def retrieve_images_from_web(num_category, num_lighting_conditions, startFrom=1):
    """
    Retrieve images from RawFoot DB and save them to images/ directory
    :param num_category: Number of image category
    :param num_lighting_conditions: Number of lightning conditions
    :return: None
    """
    image_directory = "images/"
    categories = xrange(1, num_category + 1)
    lighting_conditions = xrange(startFrom, num_lighting_conditions + 1)
    for category in categories:
        print category
        for lightning_condition in lighting_conditions:
            filename = "00%02d-%02d.png" % (category, lightning_condition)
            f = open(image_directory + filename, 'wb')
            f.write(urllib.urlopen('http://projects.ivl.disco.unimib.it/minisites/rawfoot/RawFooT_images/' + filename).read())
            f.close()

def global_adaptive_thresholding(im):
    block_size = 35
    binary_adaptive = threshold_adaptive(im, block_size, offset=10)
    return binary_adaptive

def sobel_edge_detection(im):
    """
    Get edge detection from input image
    :param im: rgb image
    :return: edge detection from rgb image
    """
    im_greyscale = img_as_float(color.rgb2gray(im))
    return sobel(im_greyscale)


def canny_edge_detection(im, sigma=1.0):
    """
    Get edge detection from input image using global thresholding
    :param im: rgb image
    :return: edge detection from rgb image
    """
    im_greyscale = img_as_float(color.rgb2gray(im))

    global_thresh = threshold_otsu(im_greyscale)
    binary_global = im_greyscale > global_thresh

    return feature.canny(binary_global, sigma)



def roberts_edge_detection(im):
    """
    Get edge detection from input image
    :param im: rgb image
    :return: edge detection from rgb image
    """
    im_greyscale = img_as_float(color.rgb2gray(im))
    return roberts(im_greyscale)


def preprocessing():
    basedir = './'
    algos = ['canny', 'roberts', 'sobel']
    # algos = ['canny']
    slash = '/'
    imagesrc = './images/'
    imagesrc = glob.glob(imagesrc + '*.png')
    imagesrc = np.sort(imagesrc)

    # algo = algos[0]
    # directory = basedir + algo + slash

    for srcim in imagesrc:
        im = canny_edge_detection(misc.imread(srcim))
        specificimname = srcim[-11:]
        plt.imsave('./canny/'+ algo + '-' + specificimname, im, cmap='Greys')

    algo = algos[1]
    directory = basedir + algo + slash

    for srcim in imagesrc:
        im = roberts_edge_detection(misc.imread(srcim))
        specificimname = srcim[-11:]
        plt.imsave('./' + algo + '/'+ algo + '-' + specificimname, im, cmap='Greys')

    algo = algos[2]
    directory = basedir + algo + slash

    for srcim in imagesrc:
        im = sobel_edge_detection(misc.imread(srcim))
        specificimname = srcim[-11:]
        plt.imsave('./' + algo + '/' + algo + '-' + specificimname, im, cmap='Greys')




if __name__ == "__main__":
    print 'hi'
    retrieve_images_from_web(10, 20)
    retrieve_images_from_web(10, 46, 44)
    preprocessing()