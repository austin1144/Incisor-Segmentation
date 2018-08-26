import numpy as np
from Figure_denoise import median_filter, crop, sobel, canny, bilateral_filter, top_hat_transform, bottom_hat_transform
import cv2
import numpy as np
# all picture

def cut_a_box(x_box_start,x_box_length,y_box_start, y_box_length ):
    """
    define a box, up left corner is the start point
    :param x_box_start:
    :param x_box_length:
    :param y_box_start:
    :param y_box_length:
    :return: the image region within the box
    """
    rect = img[y_box_start: (y_box_start + y_box_length), x_box_start:(x_box_start + x_box_length)]  # [y, x]
    return rect

def cal_mean_within_box(img):
    total = 0
    for i in range(np.shape(img)[0]):
        for j in range (np.shape(img)[1]):
            total += img[i, j]
    mean = total/float((np.shape(img)[0]*np.shape(img)[0]))
    #mean = total/float(2500.0)
    return mean

def first_the_first_level_rect(crop_length, x_box_length, y_box_length):
    """

    :param crop_length:
    :param x_box_length:
    :param y_box_length:
    :return: the min block top left point index from the ORIGINAL image
    """
    rect_firt = []
    crop_length
    #x_box_start = 200
    x_box_length
    #y_box_start = 200
    y_box_length
   # mean = np.zeros((((crop_length - x_box_length)/10), ((crop_length - y_box_length)/10)))
    mean = np.zeros(((crop_length - x_box_length)/10, (crop_length - y_box_length)/10))
    index_min_mean = np.zeros((14, 2))
    real_index_min_mean = np.zeros((14, 2))
    for picture in range(14):
        source = 'Data\Radiographs\%02d.tif' % (picture + 1)
        raw_img = cv2.imread(source, 0)
        cropped_img = crop(raw_img, 1000, 500, 1000, 1000)
        img = median_filter(cropped_img)
        img = bilateral_filter(img)
        #print raw_img
        img = top_hat_transform(img)
        img = bottom_hat_transform(img)
        img = sobel(img)

       # print (np.shape(img))
        for i in range(0, np.shape(img)[0] - x_box_length, 10):# x range search
            for j in range(0, np.shape(img)[1] - y_box_length, 10):# y range search
                #print (j)
                img_tmp = img[j: (j + y_box_length), i:(i + x_box_length)]
               # print (j, j+y_box_length, i, i + x_box_length)
                #print(i, j)
               #mean_tmp = float(cal_mean_within_box(img_tmp))
                mean_tmp = float(np.mean(img_tmp))
                #print(i,j)
                mean[i/10, j/10] = mean_tmp
        #print np.shape(mean)
        #print(picture)
        #print mean
        #print()
        index_min_mean[picture] = np.where(mean == np.amin(mean, axis = (0,1)))
        real_index_min_mean[picture] = index_min_mean[picture]*10
        #print index_min_mean
        rect_tmp_for_show = img[int(index_min_mean[picture][1]) + 200: int((index_min_mean[picture][1]+200 + y_box_length)),
                           int(index_min_mean[picture][0])+ 200:int((index_min_mean[picture][0]+200 + x_box_length))]
        rect_firt.append((img[int(index_min_mean[picture][1]) + 200: int((index_min_mean[picture][1]+200 + y_box_length)),
                           int(index_min_mean[picture][0])+ 200:int((index_min_mean[picture][0]+200 + x_box_length))]))
       # cv2.imshow('min_mean area', rect_tmp_for_show)
        #cv2.waitKey(500)
        #cv2.imwrite('Data\Configure\init_guess_image-%d.tif' % picture, rect_tmp_for_show)
    return rect_firt, real_index_min_mean

def the_second_level_rect(rect,x_box_length, y_box_length, min_index_for_first_level):
    off_set_y = 150
    off_set_x = 40
    rect_second = []
    index_min_mean = np.zeros((14, 2))
    real_index_min_mean = np.zeros((14, 2))
    mean = np.zeros((int((np.shape(rect)[1] - x_box_length)/10), int((np.shape(rect)[2] - y_box_length)/10)))
    for picture in range(np.shape(rect)[0]):
        for i in range(0, np.shape(rect)[1] - x_box_length, 10):  # x range search
            for j in range(0, np.shape(rect)[2] - y_box_length, 10):  # y range search
                # print (j)
                img_tmp = rect[picture][j: (j + y_box_length), i:(i + x_box_length)]
                # print (j, j+y_box_length, i, i + x_box_length)
                # print(i, j)
                # mean_tmp = float(cal_mean_within_box(img_tmp))
                mean_tmp = float(np.mean(img_tmp))
                # print(i,j)
                mean[i / 10, j / 10] = mean_tmp
        # print np.shape(mean)
        # print(picture)
        # print mean
        # print()
        index_min_mean[picture] = np.where(mean == np.amin(mean, axis=(0, 1)))
        real_index_min_mean[picture] = index_min_mean[picture]*10
        rect_tmp_for_show = rect[picture][int(index_min_mean[picture][1]+off_set_y):int((index_min_mean[picture][1] + y_box_length+ off_set_y)),
                                          int(index_min_mean[picture][0]+off_set_x):int((index_min_mean[picture][0]+ x_box_length)+off_set_x)]
        rect_second.append((rect[0][int(index_min_mean[picture][1]+off_set_y): int((index_min_mean[picture][1] + y_box_length + off_set_y)),
                                    int(index_min_mean[picture][0]+off_set_x): int((index_min_mean[picture][0] + x_box_length)+off_set_x)]))
        cv2.imshow('min_mean area', rect_tmp_for_show)
        cv2.waitKey(500)
        cv2.imwrite('Data\Configure\init_guess_second_level-%d.tif' % picture, rect_tmp_for_show)
    return rect_second, real_index_min_mean

def find_the_median_point_of_rect(rect):
    print np.shape(rect)[1]
    median_point = np.zeros((np.shape(rect)[0], 2))
    for picture in range(np.shape(rect)[0]):
        median_point[picture] = [np.shape(rect)[1]/2, np.shape(rect)[2]/2]
    print median_point
    return median_point


if __name__ =='__main__':
    #real_start_point = np.zeros(14)
    first_rect ,min_index_for_first_level= first_the_first_level_rect(crop_length =1000, x_box_length = 500, y_box_length=500)
    #print min_index_for_first_level
    #print first_rect[0]
    second_rect, min_index_for_second_level = the_second_level_rect(first_rect, 440, 200, min_index_for_first_level)
   #print min_index_for_second_level
    real_start_point = min_index_for_first_level + min_index_for_second_level
    print(find_the_median_point_of_rect)
    middel_point = [real_start_point[0] + find_the_median_point_of_rect(second_rect)[0],
                    real_start_point[1] + find_the_median_point_of_rect(second_rect)[1]]



    print middel_point
    #print(np.shape(first_rect))




    #print index_min_mean





