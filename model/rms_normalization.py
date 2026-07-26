import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        gamma = np.array(gamma)
        # RMS = sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.mean(x ** 2) + eps)
        # normalize
        x_hat = x / rms
        # scale (no shift - beta)
        out = gamma * x_hat
        return np.round(out, 4).tolist()
