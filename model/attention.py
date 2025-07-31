import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class Attention(nn.Module):
    """Attention mechanism for processing sequences.
    This module computes attention weights and applies them to the input sequences."""
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        ### Edit for TensoreRT
        # create mask based on the sentence lengths
        # Vectorized mask creation
        device = attentions.device
        # lengths: (batch_size,) - ensure it's a tensor on the correct device and dtype
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=device)
        else:
            lengths = lengths.to(device)
        # Create a mask of shape (batch_size, max_len)
        idxs = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        mask = (idxs < lengths.unsqueeze(1)).float()  # (batch_size, max_len)
        
        ### End Edit for TensoreRT
        ### Old code
        # # apply mask and renormalize attention scores (weights)
        # masked = attentions * mask
        # _sums = masked.sum(-1, keepdim=True)  # sums per row

        # attentions = masked.div(_sums + 1e-8)  # add epsilon to avoid division by zero

        # mask = torch.ones(attentions.size()).to(attentions.device)
        # for i, l in enumerate(lengths):  # skip the first sentence
        #     if l < max_len:
        #         mask[i, l:] = 0
        # mask.requires_grad = True
        ### End Old code

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer models."""
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
