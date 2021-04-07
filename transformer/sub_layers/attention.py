import torch
from torch import nn

from transformer.attentions.multi_head_attention import MultiHeadAttention


class Attention(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, head: int, p: int):
        """attention sub layer 구현 클래스

        Args:
            d_model (int): input, output dim
            d_k (int): key, query dim
            d_v (int): value dim
            head (int): parallel attention layers
            p (int): dropout probability
        """

        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model,
                                            d_k=d_k,
                                            d_v=d_v, 
                                            head=head)

        self.dropout = nn.Dropout(p=p)

        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, v, k, q, mask=None):
        """forward 함수

        Args:
            v (torch.Tensor(bs, len_k, d_model)): value
            k (torch.Tensor(bs, len_k, d_model)): key
            q (torch.Tensor(bs, len_q, d_model)): query
            mask (torch.Tensor(bs, 1, len_q, len_k)): mask

        Returns:
            output (torch.Tensor(bs, len_q, d_model)): forward 출력값
        """

        residual = q

        x = self.attention(v, k, q, mask)

        x = self.dropout(x)

        output = self.norm(x + residual)

        return output



