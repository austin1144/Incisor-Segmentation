
import numpy as np
from GPA import gpa
from Plotter import plot_PCA



class ASM(object):
    """Class representing an active shape model.

    Attributes:
        mean_shape (Landmarks): The mean shape of this model.
        pc_modes (np.ndarray(n,p)): A numpy (n,p) array representing the PC modes
            with p the number of modes an n/2 the number of shape points.

    """




    def __init__(self, landmarks):
        """Build an active shape model from the landmarks given.

        Args:
            landmarks (Landmarks): The landmark points from which the ASM is learned.

        """

        # Do Generalized Procrustes analysis
        mu, Xnew, mu_value = gpa(landmarks) #  mean_shape, aligned_shapes, mean_shape_value

        # covariance calculation
        XnewVec = as_vectors(Xnew)
        S = np.cov(XnewVec, rowvar=0)
        self.mu_value  = mu_value
        self.k = len(mu_value)      # Number of points
        self.mean_shape = mu    # the mean 14shape of 40 points
        self.covariance = S
        self.aligned_shapes = Xnew

        # PCA on shapes
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(-eigvals)   # Ensure descending sort
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.scores = np.dot(XnewVec, eigvecs)
        self.mean_scores = np.dot(mu.as_vector(), eigvecs)
        self.variance_explained = np.cumsum(eigvals/np.sum(eigvals))

        # Build modes for up to 98% variance
        def index_of_true(arr):
            for index, item in enumerate(arr):
                if item:
                    return index, item
        npcs, _ = index_of_true(self.variance_explained > 0.99)
        npcs += 1
        self.eigvecs = eigvecs[:, :npcs]
        #==========plot eigen teeth
        # plot_PCA(self.mu_value) # to plot the mean value
        # self.eigvecs = eigvecs[:, 6]
        # projections = np.dot(eigvals, self.eigvecs)
        # eigentooth = np.dot(projections, self.eigvecs.T).T
        # eigentooth.resize(40, 2)
        # eigentooth += self.mu_value
        # plot_PCA(eigentooth)
        # print "eigentooth done"
        #=====from internet
        # const = np.dot(self.scores, eigvecs.T)[2,:].reshape(40,2)+self.mean_scores #13,80
        # plot_PCA(const)



        M = []
        for i in range(0, npcs-1):
            M.append(np.sqrt(eigvals[i]) * eigvecs[:, i])
        self.pc_modes = np.array(M).squeeze().T

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