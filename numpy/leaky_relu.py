import numpy as np


def leaky_relu(x: ndarray) -> ndarray:
        """Apply leaky relu to an ndarray"""
        return np.maximum(0.2*x, x)


def square(x: ndarray) -> ndarray:
        """Apply squares to the elements of an ndarray"""
        return np.power(x, 2)
