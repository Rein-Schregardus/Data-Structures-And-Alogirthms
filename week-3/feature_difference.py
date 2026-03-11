from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def order_by_number(dataset_images, dataset_labels):
    ordered_images = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for _, i in enumerate(dataset_images):
        ordered_images[dataset_labels[i]].append(dataset_images[i])
    return ordered_images


(images, labels), (_,_) = mnist.load_data()
ordered_images = order_by_number(images, labels)


def calculate_average_differances(image):
    likeness  = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for (number, comparers) in ordered_images:
        for compareable in comparers:
            likeness[number] += np.absolute(image  - compareable)
    return likeness