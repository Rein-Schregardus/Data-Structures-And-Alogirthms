import numpy as np

def normalize_and_vectorize_image(image: np.array):
    """
    Turn a image of variable dimesions (must be square) into a vector and normalize values between 0 and 1
    """
    flat = image.flatten()

    maximum = np.max(flat)
    normalised: np.array = flat / maximum

    return normalised.astype(np.half)