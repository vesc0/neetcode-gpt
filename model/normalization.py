import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)

        eps = 1e-5 # prevents division by zero
        mean = np.mean(x) 
        var = np.var(x)

        x_norm = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_norm + beta # scale and shift
        
        return np.round(out, 5)
