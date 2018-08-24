import sys
import os
import hashlib
import cv2
import numpy as np



def load(path, indices):
    """Loads a series of radiograph images.

    Args:
        path: the path where the radiographs are stored.
        indices: indices of the radiographs which should be loaded.

    Returns:
        An array with the requested radiographs as 3-channel color images,
        ordered the same as the given indices.
    """

    # load images into images array
    files = ["\%02d.tif" % i if i < 15 else "\extra\%02d.tif" % i for i in indices]
    files = [path + f for f in files]
    images = []
    for f in files:
        print(f)
        images.append(cv2.imread(f))
    """
    # help debug
        print(images)
    print(np.shape(images))
    cv2.imshow('wkuec', images[1])
    cv2.waitKey(0)
    # check if all loaded files exist
    for index, img in zip(indices, images):
        if img is None:
            raise IOError("%s\%02d.tif does not exist" % (path, index))
    """
    return images


def clahe(img):
    """Creates a CLAHE object and applies it to the given image.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The result of applying CLAHE to the given image.

    """
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)

def bilateral_filter(img):
    """Applies a bilateral filter to the given image.
    This filter is highly effective in noise removal while keeping edges sharp.

    Args:
        img: A grayscale dental x-ray image.

    Returns:
        The filtered image.

    """
    return cv2.bilateralFilter(img, 9, 175, 175)  # Only this one. repeatable  Gaussian variance + grey scale


def sobel(img):
    #preprocess
    #print[img]
    #i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print i
    #i = bilateral_filter(img)
    i = clahe(img)
    i = median_filter(i)
    #i = img
    cv2.imshow('clahe', i)
    #use sobel to get edges
    sobelx = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def median_filter(img):
    img = cv2.medianBlur(img,5)
    return img

def crop(img,x,y,h, w):
    crop_img = img[y:y + h, x:x + w]
    return crop_img

def canny(img):
   return cv2.Canny(img, 30, 45)



if __name__ == '__main__':
    img = load(path='Data\Radiographs', indices=range(1, 15))[0]
    img_bilater = bilateral_filter(img)
    img_clahe = clahe(img)
    cv2.imshow('bilateral_filter',  bilateral_filter(img))
    cv2.imshow('clahe', clahe(img))
    cv2.waitKey(0)
    cv2.imwrite('Data/Configure/bilateral_filter.tif', img_bilater)
    cv2.imwrite('Data/Configure/raw.tif', img)

