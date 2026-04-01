import math;
from random import choices;
import numpy as np;
import matplotlib.pyplot as plt;


def order_by_number(dataset_images, dataset_labels):
    """
    Seperate the mnist dataset into a dictionary ordered by label
    """
    ordered_images = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for i, _ in enumerate(dataset_images):
        ordered_images[dataset_labels[i]].append(dataset_images[i])
    return ordered_images


def normalize_and_vectorize_image(image: np.array):
    """
    Turn a image of variable dimesions (must be square) into a vector and normalize values between 0 and 1
    """
    flat = image.flatten()

    maximum = np.max(flat)
    normalised: np.array = flat / maximum

    return normalised.astype(np.half)



def dist(a: np.array, b: np.array):
    """
    calculate the distance between two points in N-dimensional space
    """
    abs_differances = np.abs(a - b)
    dist = math.sqrt(np.sum(abs_differances**2))
    return dist


def k_means(datapoints: tuple[np.array], clusters_amount: int) -> set[tuple]:
    """
    Find the cluster centers
    """
    clusterOrigins: np.array[np.array] = np.array(choices(datapoints, k=clusters_amount), dtype=np.half)
    new_origins = list()
    while True:

        clusters = dict()

        # reassign points to cluster_origins
        for point in datapoints:
            closest_cluster = min(clusterOrigins, key=lambda c : dist(c, point))

            closest_cluster = closest_cluster.tobytes()
            if (closest_cluster not in clusters.keys()):
                clusters[closest_cluster] = list()

            clusters[closest_cluster].append(point)

        # recalcuate origins
        new_origins.clear()
        for key in clusters:
            mean = np.mean(clusters[key], axis=0)
            new_origins.append(mean)

        # base case
        if np.array_equal(np.array(new_origins, dtype=np.half), clusterOrigins):
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

def make_prototype(train_images, train_labels, prototypes_per_number: int):
    ordered_images = order_by_number(train_images, train_labels)
    prototype: dict[int: np.array] = dict()
    for num in ordered_images:
        prototype[num] = k_means(ordered_images[num], prototypes_per_number)
        print(f"finished for number {num}")
    print("finished computing prototypes")
    return prototype

def get_distance(a: np.array, b: np.array):
    return dist(a, b)

def predict(image, prototypes):
    img_vector = normalize_and_vectorize_image(image)
    likeness  = {0: 999999999, 1: 999999999, 2: 999999999, 3: 999999999, 4: 999999999, 5: 999999999, 6: 999999999, 7: 999999999, 8: 999999999, 9: 999999999}
    for number in prototypes:
        prototypes_values = prototypes[number]
        for prototype in prototypes_values:
            differance = dist(prototype, img_vector)
            if likeness[number] > differance:
                likeness[number] = differance
    return min(likeness, key=lambda k: likeness[k])

def determine_accuracy(prototypes, images, labels):
    total = len(images)
    misses = 0
    miss_for_num: dict  = {0: 0, 1: 0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for i, image in enumerate(images):
        prediction = predict(image, prototypes)
        # print(f"pred: {prediction} | act: {labels[i]}")
        if prediction != labels[i]:
            misses += 1
            miss_for_num[labels[i]] += 1
    print(f"misses {misses}")
    print(f"accuracy {(total - misses) / total * 100} %")
    print(miss_for_num)