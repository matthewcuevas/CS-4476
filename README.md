# CS-4476

## preprocessing.py
This will get 23 images for 10 textures: the first 20 and then the red, green, and blue. 230 images total.

It runs canny edge detection with sigma = 1, roberts, and sobel detection. 

Canny edge detection is further pre-processed using global thresholding. Learn more here: http://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_threshold_adaptive.html and https://en.wikipedia.org/wiki/Otsu's_method and http://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_local_otsu.html

It will then preprocess these photos into three folders: /canny, /roberts, /sobel.
