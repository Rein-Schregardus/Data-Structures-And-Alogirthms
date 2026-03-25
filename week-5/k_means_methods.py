import math;
from random import choices;
import numpy as np;
import matplotlib.pyplot as plt;


def order_by_number(dataset_images, dataset_labels):
    """
    Seperate the mnist dataset into a dictionary ordered by label
    """
    ordered_images = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    print(range(len(dataset_images)))
    for i, _ in range(len(dataset_images)):
        ordered_images[dataset_labels[i]].append(dataset_images[i])
    print(ordered_images)
    return ordered_images


def normalize_and_vectorize_image(image):
    """
    Turn a image of variable dimesions (must be square) into a vector and normalize values between 0 and 1
    """
    flat = image.flatten()

    maximum = np.max(flat)
    normalised = flat / maximum

    return normalised



def dist(a: np.array, b: np.array):
    """
    calculate the distance between two points in N-dimensional space
    """
    abs_differances = np.abs(a - b)
    dist = math.sqrt(np.sum(abs_differances**2))
    return dist




def k_means(datapoints: tuple[np.array], clusters: int) -> set[tuple]:
    """
    Find the cluster centers
    """
    clusterOrigins: np.array[np.array] = np.array(choices(datapoints, k=clusters))
    new_origins: np.array[np.array]
    while True:

        clusters = dict()

        # reassign points to cluster_origins
        for point in datapoints:
            closest_cluster = min(clusterOrigins, key=lambda c : dist(c, point))
            if (closest_cluster not in list(clusters.keys())):
                clusters[closest_cluster] = list()

            clusters[closest_cluster].append(point)

        # recalcuate origins
        new_origins = set()
        for key in clusters:
            mean = np.mean(clusters[key])
            new_origins.add((mean))

        # base case
        if np.array_equal(new_origins, clusterOrigins):
            return new_origins
        clusterOrigins = new_origins

def vector_to_image(vector: np.array):
    """
    Turn a vector into an image
    """
    length = len(vector)
    width = int(math.sqrt(length))
    square_vector = vector[:width**2]
    image = np.reshape(square_vector, (width, width))
    return image


def show_image(image):
    """
    Show a greyscale square image of any resolution. Image is stored in 2d array.
    """
    plt.imshow(image, cmap="grey")
    plt.show()
