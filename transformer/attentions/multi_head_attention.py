import torch
from torch import nn

from transformer.attentions.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, head: int):
        """multi head attention 구현 클래스

        Args:
            d_model (int): input, output dim
            d_k (int): key, query dim
            d_v (int): value dim
            head (int): parallel attention layers
        """

        super().__init__()
        # d_model = head * d_v = head * d_k
        self.d_model, self.d_k, self.d_v, self.head = d_model, d_k, d_v, head
        self.v_linear = nn.Linear(in_features=d_model, out_features=head * d_v, bias=False)
        self.k_linear = nn.Linear(in_features=d_model, out_features=head * d_k, bias=False)
        self.q_linear = nn.Linear(in_features=d_model, out_features=head * d_k, bias=False)

        self.attention = ScaledDotProductAttention()
        
        self.out_linear = nn.Linear(in_features=head * d_v, out_features=d_model, bias=False)

    def forward(self, 
                v: torch.Tensor, 
                k: torch.Tensor,
                q: torch.Tensor, 
                mask :torch.Tensor=None):
        """forward 함수

        Args:
            v (torch.Tensor(bs, len_k, d_model)): value
            k (torch.Tensor(bs, len_k, d_model)): key
            q (torch.Tensor(bs, len_q, d_model)): query
            mask (torch.Tensor(bs, len_q, len_k)): masking idx
        
        Returns:
            output (torch.Tensor(bs, len_q, d_model)): forward 결과값
        """

        bs,  len_k, len_q = k.size(dim=0), k.size(dim=1), q.size(dim=1)
        v_linear = self.v_linear(v).view(bs, len_k, self.head, self.d_v).transpose(1, 2)
        k_linear = self.k_linear(k).view(bs, len_k, self.head, self.d_k).transpose(1, 2)
        q_linear = self.q_linear(q).view(bs, len_q, self.head, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.head, 1, 1)

        attention = self.attention(q_linear, k_linear, v_linear, mask)
        attention = attention.transpose(1, 2).contiguous().view(bs, len_q, self.head * self.d_v)

        output = self.out_linear(attention)

        return output
        

