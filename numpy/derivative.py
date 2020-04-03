from typing import Callable

def deriv(func: Callable[[ndarrray], ndarray],
                input_: ndarray,
                delta: float = 0.001) -> ndarray:
        """
        Evaluates the derivatives of a function "func" at every element in the input array"""
        return (func(input_ + delta) - func(input_ - delta)) / (2*delta)
