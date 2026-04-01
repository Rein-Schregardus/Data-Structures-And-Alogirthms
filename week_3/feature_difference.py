from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def order_by_number(dataset_images, dataset_labels):
    ordered_images = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for i, _ in enumerate(dataset_images):
        ordered_images[dataset_labels[i]].append(dataset_images[i])
    return ordered_images

def make_standard_devs():
    standard_images = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for i, _ in enumerate(ordered_images):
        standard_images[i] = np.mean(images, axis=0)
    return ordered_images

(images, labels), (_,_) = mnist.load_data()
ordered_images = order_by_number(images, labels)
stdevs = make_standard_devs()

# this method proves to be very ineffective at identifying numbers
def calculate_average_differances(image):
    likeness  = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for number in ordered_images:
        comparers = ordered_images[number]
        for compareable in comparers:
            likeness[number] += np.sum(np.absolute(compareable - image))
    return likeness

def calculate_min_differances(image):
    likeness  = {0: 999999999, 1: 999999999, 2: 999999999, 3: 999999999, 4: 999999999, 5: 999999999, 6: 999999999, 7: 999999999, 8: 999999999, 9: 999999999}
    for number in ordered_images:
        comparers = ordered_images[number]
        for compareable in comparers:
            differance = np.sum(np.absolute(image - compareable))
            if likeness[number] > differance:
                likeness[number] = differance
    return likeness

# def calculate_stdev_differances(image):
#     likeness  = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
#     for num in stdevs:
#         likeness[num] = np.sum(np.absolute(stdevs[num] - image))
#     return likeness