import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided. It returns raw logits (not probabilities).
# Implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ix = torch.randint(len(data) - context_length, (batch_size,))
            x = torch.stack([data[i:i + context_length] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])

            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Return the final loss rounded to 4 decimals.
        return round(loss.item(), 4)
