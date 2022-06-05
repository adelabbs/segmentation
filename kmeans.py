#!/usr/bin/env python

# ---------------------------------
# CS472 - Assignment 5
# Date: 05/06/2022
# Adel Abbas, csdp1270
#
# Run:
#   python3 kmeans.py
# ---------------------------------

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import loadmat


def readImage(filename):
    return cv.imread(filename, cv.IMREAD_COLOR)


def writeImage(img, filename):
    cv.imwrite(filename, img)
    return None


def computeDistanceMatrix(A, B):
    """
    In order to benefit from the efficiency of vectorized numpy computations,
    the distance matrix dist(A, B) is computed as :
    dist(A, B) = sqrt(dot(A, A) + dot(B, B) - 2 * dot(A, B))
    """
    M = A.shape[0]
    N = B.shape[0]
    AA = np.sum(A*A, axis=1).reshape((M, 1)) * np.ones((1, N))
    BB = np.sum(B*B, axis=1) * np.ones((M, 1))
    distances = AA + BB - 2 * A.dot(B.T)
    distances = np.absolute(distances)
    return np.sqrt(distances)


def kmeans(data, k):

    # We assume that the data matrix stores n-dimensional data items as its column vectors
    # However, it's easier to work with n-dimensional data items as rows
    data = np.array(data.T, dtype=np.float32)
    labels = np.zeros(data.shape[0], dtype=np.uint8)
    # Randomly initialize cluster centers :
    # Here, we randomly select k data points to be the initial cluster centers
    idx = np.random.choice(data.shape[0], size=k, replace=False)
    means = np.array(data[idx, :], dtype=np.float32)

    converged = False
    PRECISION = 0.00001
    NB_ITER = 300
    VERBOSE = False
    iter = 0
    while not converged:
        # Assign labels
        distances = computeDistanceMatrix(data, means)
        labels = np.argmin(distances, axis=1)

        # Recompute cluster centers
        # If no data point is assigned to a given cluster (rare corner case)
        # its center is reassigned to a random data point
        new_means = np.array([np.mean(data[labels == i], axis=0)
                              if np.count_nonzero(labels == i) > 0 else data[np.random.choice(data.shape[0])]
                              for i in range(k)])

        curr_diff = np.linalg.norm(means - new_means)
        converged = curr_diff <= PRECISION
        means = new_means

        if VERBOSE and iter % 10 == 0:
            print(f"current iter: {iter}")
            print(
                f"distance between current and previous centers : {curr_diff}")
        if iter >= NB_ITER:
            break

        iter += 1
    return labels, means.T


def imSegment(im, k):
    """
    Performs k-means based segmentation using the LUV pixel values
    """
    luv = cv.cvtColor(im, cv.COLOR_RGB2Luv)
    vectors = luv.reshape((luv.shape[0] * luv.shape[1], 3)).T

    labels, means = kmeans(vectors, k)

    # map each image luv value to its corresponding cluster center
    segmented = np.array([means[:, labels[i]]
                         for i in range(vectors.shape[1])], dtype=np.uint8)

    segmented = segmented.reshape((luv.shape))

    # Convert the luv image back to rbg
    rgb = cv.cvtColor(segmented, cv.COLOR_Luv2RGB)
    return rgb


def imSegmentRGB(im, k):
    """
    Perform k-means based segmentation on the RBG image values
    """
    vectors = im.reshape((im.shape[0] * im.shape[1], 3)).T
    labels, means = kmeans(vectors, k)
    segmented = np.array([means[:, labels[i]]
                         for i in range(vectors.shape[1])], dtype=np.uint8)
    segmented = segmented.reshape((im.shape))
    return segmented


def imSegmentWithPosition(im, k):
    """
    Performs k-means based segmentation using both LUV and pixel coordinates
    """
    luv = cv.cvtColor(im, cv.COLOR_RGB2Luv)
    vectors = luv.reshape((luv.shape[0] * luv.shape[1], 3)).T

    # add (x,y) image coordinates
    row, col = np.indices((luv.shape[0], luv.shape[1]))
    row = row.reshape((luv.shape[0] * luv.shape[1], 1)).T
    col = col.reshape((luv.shape[0] * luv.shape[1], 1)).T
    xy = np.vstack((row, col))
    vectors = np.vstack((vectors, xy))

    labels, means = kmeans(vectors, k)

    # map each image luv value to its corresponding cluster center
    segmented = np.array([means[:3, labels[i]]
                         for i in range(vectors.shape[1])], dtype=np.uint8)

    segmented = segmented.reshape((luv.shape))

    # Convert the luv image back to rbg
    rgb = cv.cvtColor(segmented, cv.COLOR_Luv2RGB)
    return rgb


def testKMeans():
    DIR = "./data"
    filename = "pts.mat.mat"
    mat = loadmat(os.path.join(DIR, filename))
    data = np.array(mat["data"])
    for k in [1, 2, 4, 8]:
        labels, means = kmeans(data, k)
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(data[0, :], data[1, :], data[2, :], c=labels)
        plt.title(f"k-means classification with k = {k}")
        plt.show()

    return None


def main():
    #testKMeans()
    DIR = "./data"
    OUT = "./out"
    if not os.path.exists(OUT):
        os.makedirs(OUT)

    directory = os.fsencode(DIR)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("bmp"):
            im = readImage(os.path.join(DIR, filename))
            for k in [3, 5, 7, 10]:
                # k-means LUV
                print(f"luv k-means segmentation of {filename} with k={k}...")
                seg = imSegment(im, k)
                writeImage(seg, os.path.join(
                    OUT, filename+"_luv_kmeans_" + str(k) + ".bmp"))

                # k-means RGB
                print(f"rgb k-means segmentation of {filename} with k={k}...")
                seg = imSegmentRGB(im, k)
                writeImage(seg, os.path.join(
                    OUT, filename+"_rgb_kmeans_" + str(k) + ".bmp"))

            # k-means LUV + (x,y)
            for k in [21, 35, 50]:
                print(
                    f"luv + xy k-means segmentation of {filename} with k={k}...")
                seg = imSegmentWithPosition(im, k)
                writeImage(seg, os.path.join(
                    OUT, filename+"_xy_luv_kmeans_" + str(k) + ".bmp"))

    return None


if __name__ == "__main__":
    main()
