import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        """scaled dot product attention 구현 클래스"""

        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor=None):
        """forward 함수

        Args:
            q (torch.Tensor(bs, head, len_q, d_k)): query
            k (torch.Tensor(bs, head, len_k, d_k)): key
            v (torch.Tensor(bs, head, len_k, d_v)): value
            mask (torch.Tenso(bs, 1, len_q, len_k)): masking idx

        Returns:
            output (torch.Tensor(bs, head, len_q, d_v)): forward 결과값
        """

        d_k = k.size(dim=-1)

        weight = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            weight.masked_fill(mask==False, -1e12)

        scale_weight = self.softmax(weight)

        output = scale_weight @ v

        return output