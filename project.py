import glob
import urllib

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage import feature, img_as_float, color
from skimage.filters import roberts, sobel, threshold_otsu, threshold_adaptive
from sklearn import tree, neural_network, neighbors, metrics
from sklearn.datasets.base import load_data
from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition.pca import PCA

RANDOM_SEED = 37219857

def retrieve_images_from_web(num_category, num_lighting_conditions):
    """
    Retrieve images from RawFoot DB and save them to images/ directory

    :param num_category: Number of image category
    :param num_lighting_conditions: Number of lightning conditions
    :return: None
    """
    print "Retrieving images from RawFoot DB..."
    image_directory = "images/"
    categories = xrange(1, num_category + 1)
    lighting_conditions = xrange(1, num_lighting_conditions + 1)
    for category in categories:
        for lightning_condition in lighting_conditions:
            filename = "00%02d-%02d.png" % (category, lightning_condition)
            f = open(image_directory + filename, 'wb')
            f.write(urllib.urlopen('http://projects.ivl.disco.unimib.it/minisites/rawfoot/RawFooT_images/' + filename).read())
            f.close()
    print "Retrieving images from RawFoot DB... Done!"

def edge_detection(im):
    """
    Get edge detection from input image

    :param im: rgb image
    :return: edge detection from rgb image
    """
    im_greyscale = img_as_float(color.rgb2gray(im))
    return feature.canny(im_greyscale)

def get_dataset_from_local_storage(num_category, num_lighting_conditions, filter):
    """
    Read images from local directory and detect edges from the images.
    Split images into training data and testing data

    :param num_category: # of categories
    :param num_lighting_conditions: # of lighting conditions
    :return: training/testing data
    """
    print "Getting dataset from local storage..."
    num_images = num_category * num_lighting_conditions
    categories = xrange(1, num_category + 1)
    lighting_conditions = xrange(1, num_lighting_conditions + 1)
    image_directory = './' + filter + '/'
    index = 0
    images, labels = np.zeros((num_images, 800 * 800 * 4)), np.zeros((num_images))
    for category in categories:
        for lighting_condition in lighting_conditions:
            filename = filter + ("-00%02d-%02d.png" % (category, lighting_condition))
            # print filename
            im = misc.imread(image_directory + filename)
            im = im.flatten()
            images[index] = im
            labels[index] = category
            index += 1
    training_data, training_label, testing_data, testing_label = split_dataset(images, labels)
    print "Getting dataset from local storage... Done!"
    return training_data, training_label, testing_data, testing_label

def split_dataset(images, labels):
    """
    Randomly split dataset into training/testing set in 70/30 ratio

    :param images: input images
    :param labels: input labels
    :return: splitted dataset
    """
    training_set_size = int(len(images) * .7)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(images))
    training_indices, testing_indices = indices[:training_set_size], indices[training_set_size:]

    training_data, training_label, testing_data, testing_label = images[training_indices], labels[training_indices], images[testing_indices], labels[testing_indices]
    return training_data, training_label, testing_data, testing_label

def decision_tree(training_data, training_label, testing_data, testing_label):
    """
    Use decision tree to classify images
    """
    print "Training decision tree... "
    max_depth = range(2, 20)
    training_error = [0] * len(max_depth)
    testing_error = [0] * len(max_depth)

    for i, depth in enumerate(max_depth):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(training_data, training_label)
        training_error[i] = metrics.mean_squared_error(training_label, clf.predict(training_data))
        testing_error[i] = metrics.mean_squared_error(testing_label, clf.predict(testing_data))

    plt.figure()
    plt.title('Decision Trees: Mean Squared Error x Max Depth')
    plt.plot(max_depth, testing_error, '-', label='testing error')
    plt.plot(max_depth, training_error, '-', label='training error')
    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error')
    plt.savefig('DecisionTree.png')
    plt.close()
    print "Training decision tree... Done!"


def neural_net(training_data, training_label, testing_data, testing_label):
    """
    Use neural network to classify images
    """
    print "Training neural network... "
    num_hidden_layers = range(10, 50, 5)
    training_error = [0] * len(num_hidden_layers)
    testing_error = [0] * len(num_hidden_layers)

    for i, hidden_layer in enumerate(num_hidden_layers):
        clf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_layer, 30), random_state=1)
        clf = clf.fit(training_data, training_label)
        training_error[i] = metrics.mean_squared_error(training_label, clf.predict(training_data))
        testing_error[i] = metrics.mean_squared_error(testing_label, clf.predict(testing_data))

    plt.figure()
    plt.title('Neural Network: Mean Squared Error x Num Estimators')
    plt.plot(num_hidden_layers, testing_error, '-', label='testing error')
    plt.plot(num_hidden_layers, training_error, '-', label='training error')
    plt.legend()
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Squared Error')
    plt.savefig('NeuralNet.png')
    plt.close()
    print "Training neural network... Done!"


def k_nearest_neighbors(training_data, training_label, testing_data, testing_label):
    """
    Use K nearest neighbors to classify images
    """
    print "Training knn... "
    K = range(2, 7)
    training_error = [0] * len(K)
    testing_error = [0] * len(K)

    for i, k in enumerate(K):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf = clf.fit(training_data, training_label)
        training_error[i] = metrics.mean_squared_error(training_label, clf.predict(training_data))
        testing_error[i] = metrics.mean_squared_error(testing_label, clf.predict(testing_data))

    plt.title('KNN: Mean Squared Error x K')
    plt.plot(K, testing_error, '-', label='testing error')
    plt.plot(K, training_error, '-', label='training error')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Mean Square Error')
    plt.savefig('KNN.png')
    plt.close()
    print "Training knn... Done!"


def pca(training_data, testing_data):
    """
    Dimensionality Reduction using Principal Component Analysis

    :param training_data: training data
    :param testing_data: testing data
    :return: dimensionality-reduced training/testing data
    """
    compressor = PCA(n_components=int(len(training_data[0])/2))
    compressor.fit(training_data)
    new_training_data = compressor.transform(training_data)
    new_testing_data = compressor.transform(testing_data)
    return new_training_data, new_testing_data


def ica(training_data, testing_data):
    """
    Dimensionality Reduction using Independent Component Analysis

    :param training_data: training data
    :param testing_data: testing data
    :return: dimensionality-reduced training/testing data
    """
    compressor = ICA(whiten=False)
    compressor.fit(training_data)
    new_training_data = compressor.transform(training_data)
    new_testing_data = compressor.transform(testing_data)
    return new_training_data, new_testing_data

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
    algos = ['canny', 'roberts', 'sobel']
    imagesrc = np.sort(glob.glob('./images/*.png'))

    algo = algos[0]

    for srcim in imagesrc:
        im = canny_edge_detection(misc.imread(srcim))
        specificimname = srcim[-11:]
        plt.imsave('./canny/'+ algo + '-' + specificimname, im, cmap='Greys')

    algo = algos[1]

    for srcim in imagesrc:
        im = roberts_edge_detection(misc.imread(srcim))
        specificimname = srcim[-11:]
        plt.imsave('./' + algo + '/'+ algo + '-' + specificimname, im, cmap='Greys')

    algo = algos[2]

    for srcim in imagesrc:
        im = sobel_edge_detection(misc.imread(srcim))
        specificimname = srcim[-11:]
        plt.imsave('./' + algo + '/' + algo + '-' + specificimname, im, cmap='Greys')

def convert_to_npy(filter):
    training_data, training_label, testing_data, testing_label = get_dataset_from_local_storage(
        num_category, num_lighting_conditions, filter=filter)
    np.save(filter + '_training_data.npy', training_data)
    np.save(filter + '_training_label.npy', training_label)
    np.save(filter + '_testing_data.npy', testing_data)
    np.save(filter + '_testing_label.npy', testing_label)

def run_experiment(training_data, training_label, testing_data, testing_label, dimensionality_reduction=None):
    if dimensionality_reduction == 'PCA':
        print "Using PCA to reduce dimensionality"
        training_data, testing_data = pca(training_data, testing_data)
    elif dimensionality_reduction == 'ICA':
        print "Using ICA to reduce dimensionality"
        training_data, testing_data = ica(training_data, testing_data)
    decision_tree(training_data, training_label, testing_data, testing_label)
    neural_net(training_data, training_label, testing_data, testing_label)
    k_nearest_neighbors(training_data, training_label, testing_data, testing_label)


def save_images_to_npy(num_category, num_lighting_conditions):
    # retrieve_images_from_web(num_category, num_lighting_conditions)
    # preprocessing()
    filters = ['canny', 'sobel', 'roberts']
    for filter in filters:
        convert_to_npy(filter)


def load_npy_file(filter):
    training_data = np.load(filter + '_training_data.npy')
    training_label = np.load(filter + '_training_data.npy')
    testing_data = np.load(filter + '_testing_data.npy')
    testing_label = np.load(filter + '_testing_label.npy')

    return training_data, training_label, testing_data, testing_label

if __name__ == "__main__":
    # Run this part only once and comment it out after
    num_category, num_lighting_conditions = 2, 2
    save_images_to_npy(num_category, num_lighting_conditions)

    # Here are the codes for actual experiments
    filters = ['canny', 'sobel', 'roberts']
    for filter in filters:
        print "Experimenting with", filter, "filter"
        training_data, training_label, testing_data, testing_label = load_npy_file(filter)
        run_experiment(training_data, training_label, testing_data, testing_label)
        run_experiment(training_data, training_label, testing_data, testing_label, dimensionality_reduction='PCA')
        run_experiment(training_data, training_label, testing_data, testing_label, dimensionality_reduction='ICA')
        print "Experimenting with", filter, "filter... Done!"