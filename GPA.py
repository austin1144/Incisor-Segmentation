# -*- coding: utf-8 -*-
"""Code for shape alignment using Procrustes Analysis.
"""


import os
import fnmatch
import re
import pre_process
import numpy as np
from Plotter import drawlines
import cv2
import PCA
from pre_process import rescale_withoutangle
from pre_process import Landmarks
from pre_process import load_all_incisors_of_example
from pre_process import load
from Plotter import plot_procrustes
from Figure_denoise import crop
from Figure_denoise import median_filter
from util import load_training_data

def gpa(landmarks, Nr_incisor):
    """Performs Generalized Procrustes Analysis on the given landmark models.

    This does Procrustes Analysis, aligning each shape such that the sum of
    distances of each shape to the mean is minimised.

    Based on:
        An introduction to Active Shape Models - Appenix A

    Args:
        landmarks (List[Landmarks]): The landmarks of one incisor.

    Returns:
        mean_shape (Landmarks): An estimate of the mean shape.
        aligned_shapes (List[Landmarks]): All samples aligned with the mean shape.

    """
    aligned_shapes = list(landmarks)

    # 1: translate each example so that its centre of gravity is at the origin.
    aligned_shapes = [shape.translate_to_mass_center() for shape in aligned_shapes]

    # 2: choose one example (x0) as an initial estimate of the mean shape
    # and scale so that |x0| = 1
    x0 = aligned_shapes[0].scale_to_unit()
    mean_shape = x0
    # iterate
    while True:
        # 4: align all shapes with current estimate of mean shape
        for ind, lm in enumerate(aligned_shapes):
            aligned_shapes[ind] = align_shapes(lm, mean_shape)  # ind is index and lm is value in aligned_shapes

        # 5: re-estimate the mean from aligned shapes
        new_mean_shape = __compute_mean_shape(aligned_shapes)

        # 6: apply constraints on scale and orientation to the current estimate
        # of the mean by aligning it with x0 and scaling so that |x| = 1.
        new_mean_shape = align_shapes(new_mean_shape, x0)
        new_mean_shape = new_mean_shape.scale_to_unit().translate_center_to_origin()

        # debug
        plot_procrustes(new_mean_shape, aligned_shapes, Nr_incisor, True)

        # 7: if converged, do not return to 4
        #print((mean_shape.as_vector() - new_mean_shape.as_vector()))
        if ((mean_shape.as_vector() - new_mean_shape.as_vector()) < 0.1).all():

            break
  #  Plotter.plot_procrustes(new_mean_shape, aligned_shapes)

    mean_shape_value = mean_shape.show_points()
    return mean_shape, aligned_shapes, mean_shape_value


def align_shapes(x1, x2):
    """Aligns two mean_centered shapes.

    Since the normalisation has been already made, thus the rotation is left to be calculated.

    Scales and rotates x1 by (s, theta) so as to minimise |s*A*x1 - x2|,
    where A performs a rotation of a shape x by theta.

    Based on:
        An introduction to Active Shape Models - Appenices A & D

    Args:
        x1: The shape which will be scaled and rotated.
        x2: The shape to which x1 will be aligned.

    Returns:
        The aligned version of x1.

    """

    # get params
    _, s, theta = align_params(x1, x2)

    # align x1 with x2
    x1 = x1.rotate(theta)
    x1 = x1.scale(s)

    # project into tangent space by scaling x1 with 1/(x1.x2)
    xx = np.dot(x1.as_vector(), x2.as_vector())
    return Landmarks(x1.as_vector()*(1.0/xx))


def align_params(x1, x2):
    """Computes the optimal parameters for the alignment of two shapes.

    We wish to translate, scale and rotate x1 by (t, s, theta) so as to minimise
    |t+s*A*x1 - x2|, where A performs a rotation of a shape x by theta.

    Based on:
        An introduction to Active Shape Models - Appenix D

    Args:
        x1, x2: Two shapes with each as format [x0,x1...xn,y0,y1...yn].

    Returns:
        The optimal parameters t, s and theta to align x1 with x2.

    """
    # work in vector format and change data to vector format as hstack
    #print ('x2 us', x2.show_points())
    x1 = x1.as_vector()
    x2 = x2.as_vector()

    length_1 = len(x1)/2
    length_2 = len(x2)/2

    # make sure both shapes are mean centered for computing scale and rotation
    x1_centroid = np.array([np.mean(x1[:length_1]), np.mean(x1[length_1:])])
    x2_centroid = np.array([np.mean(x2[:length_2]), np.mean(x2[length_2:])])
    x1 = [x - x1_centroid[0] for x in x1[: length_1]] + [y - x1_centroid[1] for y in x1[length_1:]]
    x2 = [x - x2_centroid[0] for x in x2[:length_2]] + [y - x2_centroid[1] for y in x2[length_2:]]

    # a = (x1.x2)/|x1|^2
    norm_x1_sq = (np.linalg.norm(x1)**2)
    a = np.dot(x1, x2) / norm_x1_sq

    # b = sum_1->l2(x1_i*y2_i - y1_i*x2_i)/|x1|^2
    b = (np.dot(x1[:length_1], x2[length_2:]) - np.dot(x1[length_1:], x2[:length_2])) / norm_x1_sq

    # s^2 = a^2 + b^2
    s = np.sqrt(a**2 + b**2)

    # theta = arctan(b/a)
    theta = np.arctan(b/a)

    # the optimal translation is chosen to match their centroids
    t = x2_centroid - x1_centroid

    return t, s, theta


def __compute_mean_shape(landmarks):
    """Computes the mean shape.

    Args:
        landmarks ([Landmarks]): A list of landmark objects.

    Returns:
        A ``Landmarks`` object, representing the mean shape of the given shapes.

    """
    mat = []
    for lm in landmarks:
        mat.append(lm.as_vector())
    mat = np.array(mat)
    return Landmarks(np.mean(mat, axis=0))

#def evaluation(X, Goldenlm)




if __name__ == '__main__':
    Nr_incisor = 1
    s = 500
    t = [1370, 890]
    Golden_lm = load(Nr_incisor)
    Golden_lm = rescale_withoutangle(gpa(Golden_lm, Nr_incisor)[2], t, s )
    lm_objects = load_training_data(Nr_incisor)
    landmarks_pca = PCA.ASM(lm_objects)
    #print(np.shape(landmarks_pca.mu_value))
    landmarks_pca_value = rescale_withoutangle(landmarks_pca.mu_value, t, s )


    img = cv2.imread('Data/Radiographs/01.tif', 0)
    img = img.copy()
    img_withlm = drawlines(img, landmarks_pca_value)
    crop_img = crop(img_withlm, 1000, 500, 1000, 1000)


    img = cv2.resize(crop_img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('cropped_raw', crop_img)
    #sobel = sobel(crop_img)
    #cv2.imshow('sobel', sobel)
    #median = median_filter(crop_img)
    #cv2.imshow('median', crop_img)
    cv2.waitKey(0)


