from skimage import io
import matplotlib.pyplot as plot
from PIL import Image

def get_edges(im, sigma):
    from skimage import img_as_float
    from skimage.feature import canny
    from skimage.color import rgb2gray
    from autoCanny import auto_canny

    inputImage = img_as_float(im)
    inputGrayscaleImage = rgb2gray(inputImage)
    edges = canny(inputGrayscaleImage, sigma)
    return edges

def get_distance(edges1, edges2):
    from scipy.spatial.distance import directed_hausdorff

    return directed_hausdorff(edges1, edges2)

im1 = io.imread('whitepeas1.png')
im2 = io.imread('whitepeas2.png')

plot.subplot(2, 2, 1)
plot.imshow(im1)

plot.subplot(2, 2, 2)
plot.imshow(im2)

plot.subplot(2, 2, 3)
plot.imshow(get_edges(im1, 3), cmap = 'gray')

plot.subplot(2, 2, 4)
plot.imshow(get_edges(im2, 3), cmap = 'gray')
plot.show()

print(get_distance(get_edges(im1, 3), get_edges(im2, 3)))
