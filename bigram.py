from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLM(nn.Module):
    """
    Given one token, predicts the next token.
    There are other variations of this model, generally called N-Gram language model
    """

    def __init__(self, vocab_size):
        super().__init__()
        # these are logits, i.e for each token's row we get logits
        # of probabilites (need to apply softmax) of next token for the chosen token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # idx and target are of (B, T) shape, where B - batch size, T - context/block size
        # output logits will be (B, T, C) shape, where C - logits of size vocab_size
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """This function basically is our inference. Given some starting tokens, generate max_new_tokens"""
        for _ in range(max_new_tokens):
            # idx is (B, T)
            logits, _ = self.forward(idx)
            # since we want to know the next token for given input idx tokens (not for sub-sequence)
            # we only need the last predicted token out of T
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, -1)  # still (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append to next request
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
