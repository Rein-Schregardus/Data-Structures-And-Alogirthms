import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist;

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # mnist is imported to use for determining accuracy only.

def import_image(filepath: str) -> np.ndarray:
    image = Image.open(filepath).convert('L')
    return np.array(image)

def get_test_images():
    return test_images, test_labels