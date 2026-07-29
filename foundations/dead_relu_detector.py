import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        dead_fractions = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                # After each ReLU layer, compute the fraction of neurons that are dead.
                if isinstance(module, nn.ReLU):
                    # A neuron is dead if it outputs 0 for ALL samples in the batch.
                    dead = (x == 0).all(dim=0).float().mean().item()
                    dead_fractions.append(round(dead, 4))

        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.

        # No ReLU layers - nothing to fix.
        if len(dead_fractions) == 0:
            return 'healthy'

        max_frac = max(dead_fractions)
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        if max_frac > 0.5:
            return 'use_leaky_relu'

        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. 'reduce_learning_rate' - dead fraction increases with depth AND the last layer's fraction > 0.1
        if len(dead_fractions) >= 2:
            increasing = all(dead_fractions[i] < dead_fractions[i + 1] for i in range(len(dead_fractions) - 1))
            if increasing and dead_fractions[-1] > 0.1:
                return 'reduce_learning_rate'

        # 4. 'healthy' if max dead fraction < 0.1
        if max_frac < 0.1:
            return 'healthy'

        # 5. 'healthy' otherwise
        return 'healthy'
