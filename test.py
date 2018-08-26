import numpy as np
from pre_process import Landmarks
from PCA import ASM
from Plotter import drawlines
import cv2
from GPA import gpa

img = cv2.imread('Data/Radiographs/01.tif', 0)


Data_path = 'Data\Landmarks\c_landmarks\landmarks1-6.txt'

points = Landmarks(Data_path).show_points()
lm_after_gpa = gpa(points)[2]
Landmarks(lm_after_gpa).show_points()

img_tmp = img.copy()
img = drawlines(img_tmp, points)
img_gpalm = drawlines(img, points)

cv2.imshow('lm1', img)
cv2.waitKey(0)