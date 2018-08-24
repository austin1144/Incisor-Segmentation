#from __future__ import division
import os
import numpy as np
import fnmatch
import re
import math
DATA_DIR = os.path.join('.', 'Data\Landmarks\original')

class Landmarks(object):
    """Class representing the landmarks for one example incisor.

    Attributes:
        points (np.array([float, float])): The landmarks as a list of (x,y) coordinates.

    """

    def __init__(self, source):
        """Creates a new set of landmarks.

        Args:
            source (str || []): The path of the landmarks file ||
                A list with landmark points in [x_1,...,x_n, y_1,..., y_n] format.

        """
       # self.source = 'Data/Landmarks/original'
        self.points = np.array([])
        #self.centroid = np.array([])
        self.centroid = []
        if source is not None:
            # read from file
            if isinstance(source, str):
                self._read_landmarks(source)
            # read from vector
            elif isinstance(source, np.ndarray) and np.atleast_2d(source).shape[0] == 1:
                self.points = np.array((source[:len(source)/2], source[len(source)/2:])).T
            # read from matrix
            elif isinstance(source, np.ndarray) and source.shape[1] == 2:
                self.points = source
            else:
                raise ValueError("Unsupported source type for Landmarks object.")

    def show_points(self):
        #print(np.shape(self.points))
        #print(self.points)
        return self.points

    def _read_landmarks(self, input_file):
        """Processes the given input_file with landmarks.

        Args:
            input_file (str): The path of the landmarks file.

        """
        lines = open(input_file).readlines()
        points = []
        for x, y in zip(lines[0::2], lines[1::2]):
            points.append(np.array([float(x), float(y)])) # append arrays
        self.points = np.array(points) # as matrix
        #print(self.points)
        return self.points

    def as_vector(self):
        """Returns the landmark points in [x_1,...,x_n, y_1,..., y_n] format.

        Returns:
            A numpy array of landmark points as [x_1,...,x_n, y_1,..., y_n]

        """
       # print(np.hstack((self.points[:, 0], self.points[:, 1])))
        return np.hstack((self.points[:, 0], self.points[:, 1]))

    def as_matrix(self):
        """Returns the lanmark points in [[x_1, y_1], ..., [x_n, y_n]] format.

        Returns:
            A numpy array of landmark points as [[x_1, y_1], ..., [x_n, y_n]].

        """
        return self.points

    def get_centroid(self):
        """Returns the center of mass of this shape.

        Returns:
            The centroid as [x, y]
        """
        self.centroid = np.mean(self.points, axis=0)
        return np.mean(self.points, axis=0)


    def get_center(self):
        """Returns the center of this shape.
        # the center could be discussed
        Returns:
            The center as [x, y]
        """
        return [self.points[:, 0].min() + (self.points[:, 0].max() - self.points[:, 0].min())/2,
                self.points[:, 1].min() + (self.points[:, 1].max() - self.points[:, 1].min())/2]

    def get_crown(self, is_upper):
        """Returns the top part of the tooth, without the root.

        Returns:
            The crown part of the tooth.
        """
        if is_upper:
            return Landmarks(self.points[10:30, :])
        else:
            points = np.vstack((self.points[0:10, :], self.points[30:40, :]))
            return Landmarks(points)

    def translate_to_mass_center(self):
        """Translates the landmark points so that the centre of gravitiy of this
        shape is at the origin.

        """
        # centroid = np.mean(self.points, axis=0)
        centroid = self.get_centroid()
        points = self.points - centroid
        return Landmarks(points)

    def translate_center_to_origin(self):
        """Translates the landmark points so that the center of this
        shape is at the origin.

        """
        center = self.get_center()
        points = self.points - center
        return Landmarks(points)


    def scale_to_unit(self):
        """Scales each landmark point so that the norm of the shape is 1.

        """
        centroid = np.mean(self.points, axis=0)
        scale_factor = np.sqrt(np.power(self.points - centroid, 2).sum())
        points = self.points.dot(1. / scale_factor)
        return Landmarks(points)

    def translate(self, vec):
        """Translates this model to given centroid.

        Args:
            vec : (1, 2) numpy array representing the translation vector.[x, y]
        """
        points = self.points + vec
        return Landmarks(points)

    def scale(self, factor):
        """Rescales this model by the given factor.

        Args:
            factor: The scale factor.

        """
        centroid = self.get_centroid()
        points = (self.points - centroid).dot(factor) + centroid
        return Landmarks(points)

    def scale_to_bbox(self, bbox):
        """Rescales this model to fit in the given bbox.

        Args:
            bbox: The surrounding bbox.

        """
        bbox_h = bbox[1][1] - bbox[0][1]
        # bbox_w = bbox[1][0] - bbox[0][0]
        # scale_w = bbox_w / (self.points[:,0].max() - self.points[:,0].min())
        scale_h = bbox_h / (self.points[:, 1].max() - self.points[:, 1].min())
        return self.scale(scale_h)

    def scaleposition(self, factor):
        """Rescales the coordinates of this model by the given factor.
        (changes scale and position of the landmarks)

        Args:
            factor: The scale factor.
        """
        points = self.points.dot(factor)
        return Landmarks(points)

    def mirror_y(self):
        """Mirrors this model around the y axis.

        """
        centroid = self.get_centroid()
        points = self.points - centroid
        points[:, 0] *= -1
        points = points + centroid
        points = points[::-1]  # https://stackoverflow.com/questions/41430791/python-list-error-1-step-on-1-slice
        return Landmarks(points)

    def rotate(self, angle):
        """Rotates this model clockwise by the given angle.

        Args:
            angle: The rotation angle (in radians).

        """
        # create rotation matrix
        rotmat = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

        # apply rotation on each landmark point
        points = np.zeros_like(self.points)
        centroid = self.get_centroid()
        tmp_points = self.points - centroid
        for ind in range(len(tmp_points)):
            points[ind, :] = tmp_points[ind, :].dot(rotmat)
        points = points + centroid

        return Landmarks(points)

    def T(self, t, s, theta):
        """Performs a rotation by theta, a scaling by s and a translation
        by t on this model.

        Args:
            t: translation vector.
            s: scaling factor.
            theta: rotation angle (counterclockwise, in radians)
        """
        return self.rotate(theta).scale(s).translate(t)

    def invT(self, t, s, theta):
        """Performs the inverse transformation.
        used in Iteration method,
        fucntion: parameter_update

        Args:
            t: translation vector.
            s: scaling factor.
            theta: rotation angle (counterclockwise, in radians)
        """
        return self.translate(-t).scale(1/s).rotate(-theta)





def rescale_withoutangle(lm, t, s):
    lm = lm* s + t
    return lm


def load(incisor_nr):
    """Collects all example landmark models for the incisor with the given number.

    Args:
        incisor_nr : identifier of tooth

    Returns:
        A list containing all landmark models.

    """
    files = sorted(fnmatch.filter(os.listdir(DATA_DIR), "*-{}.txt".format(str(incisor_nr))),
                   key=lambda x: int(re.search('[0-9]+', x).group()))
    lm_objects = []
    for filename in files:
        lm_objects.append(Landmarks("{}/{}".format(DATA_DIR, filename)))

    return lm_objects

def load_mirrored(incisor_nr):
    """Extends the training set by including the mirrored landmarks of
    the matching incisor in the y-axis mirrored radiograph.

    Args:
        incisor_nr : identifier of tooth

    Returns:
        A list containing all landmark models.
    """
    base = load(incisor_nr)
    mirrored_incisor_nr = {1:4, 2:3, 3:2, 4:1, 5:8, 6:7, 7:6, 8:5}
    mirrored = [shape.mirror_y() for shape in load(mirrored_incisor_nr[incisor_nr])]
    return base + mirrored


def load_all_incisors_of_example(example_nr):
    """Collects all randmarks for a given radiograph.

    Args:
        example_nr : the index of a radiograph.

    Returns:
        A list with landmark models.
    """
    files = sorted(fnmatch.filter(os.listdir(DATA_DIR), "landmarks{}-*.txt".format(str(example_nr))))
#    print(files)
    lm_objects = []
    for filename in files:
        lm_objects.append(Landmarks("{}\{}".format(DATA_DIR, filename)))
    return lm_objects


def as_vectors(landmarks):
    """Converts a list of Landmarks object to vector format.

    Args:
        landmarks: A list of Landmarks objects

    Returns:
        A numpy N x 2*p array where p is the number of landmarks and N the
        number of examples. x coordinates are before y coordinates in each row.

    """
    mat = []
    for lm in landmarks:
        mat.append(lm.as_vector())
    return np.array(mat)

def find_the_normal_to_lm(lm):
    """Points should be lm normally and index is its index of value.
       Points has the shape as_vector

       return:
       slope: the slope of normal line
       middle_points: the middle point of two points
       """
    slope = np.zeros(40)
    #lm = lm.show_points()
   # middle_points = []
    for i in range(40-1):
        delta_x = lm[i+1, 0] - lm[i, 0]
        #print delta_x
        delta_y = lm[i+1, 1] - lm[i, 1]
        if delta_x == 0:
            slope[i] = 1
        else:
            slope[i] = -delta_y/delta_x
    #print slope
    return slope

"""
    if index == 0:  # first point
        delta_x = points[1] - points[0]
        delta_y = points[1].y - points[0].y
    elif index == len(points) - 1:  # last point
        delta_x = points[-1].x - points[-2].x
        delta_y = points[-1].y - points[-2].y
    else:
        delta_x = points[index + 1].x - points[index - 1]
        delta_y = points[index + 1].y - points[index - 1]
    mag = math.sqrt(delta_x ** 2 + delta_y ** 2)
    return (-delta_y / mag, delta_x / mag)
    """
if __name__ == '__main__':
    source = 'Data\Landmarks\original\landmarks1-1.txt'
    lm = Landmarks(source)
    print(load_all_incisors_of_example(14))
    #lm(source)