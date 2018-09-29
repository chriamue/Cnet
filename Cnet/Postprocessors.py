import numpy as np
import cv2

def scalemax(img, epsilon = 0.00001):
    img = img / (img.max()+epsilon)
    return img

def remove_small(img, min_pixels=200):
    '''
    source: https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
    removes small contours.
    '''
        #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = min_pixels 

    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img[output == i + 1] = 0

    return img