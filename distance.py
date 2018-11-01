from skimage import io
import matplotlib.pyplot as plot
from PIL import Image
import numpy as np

def get_edges(im, sigma):
    from skimage import img_as_float
    from skimage.feature import canny
    from skimage.color import rgb2gray

    inputImage = img_as_float(im)
    inputGrayscaleImage = rgb2gray(inputImage)
    edges = canny(inputGrayscaleImage, sigma)
    return edges

def get_distance(edges1, edges2):
    err = np.sum((edges1.astype("float") - edges2.astype("float")) ** 2) / float(edges1.shape[0] * edges2.shape[1])

    return err

im1path = 'whitepeas1.png'
im2path = 'whitepeas2.png'
im1 = io.imread(im1path)
im2 = io.imread(im2path)
# im2 = np.random.random(im1.shape)
sigma2 = 3

plot.subplot(2, 2, 1)
plot.title(im1path)
plot.imshow(im1)

plot.subplot(2, 2, 2)
plot.title(im2path)
plot.imshow(im2)

plot.subplot(2, 2, 3)
plot.title(im1path + " Edges")
plot.imshow(get_edges(im1, 3), cmap = 'gray')

plot.subplot(2, 2, 4)
plot.title(im2path + " Edges")
plot.imshow(get_edges(im2, sigma2), cmap = 'gray')
plot.show()

print(get_distance(get_edges(im1, 3), get_edges(im2, sigma2)))
