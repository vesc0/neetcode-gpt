import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value

        # forward pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        error = y_hat - y_true
        sigmoid_deriv = y_hat * (1.0 - y_hat)
        delta = error * sigmoid_deriv

        # gradients
        dL_dw = delta * x
        dL_db = delta

        return (np.round(dL_dw, 5), round(dL_db, 5))
