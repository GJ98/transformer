import torch
from torch import nn

from transformer.sub_layers.attention import Attention
from transformer.sub_layers.feed_forward import FeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, head: int, d_ff: int, p: int):
        """encoder layer 구현 클래스

        Args:
            d_model (int): intput, output dim
            d_k (int): key, query dim
            d_v (int): value dim
            head (int): parallel attention layers
            d_ff (int): hidden dim
            p (int): dropout probability
        """

        super().__init__()
        self.attention = Attention(d_model=d_model,
                                   d_k=d_k,
                                   d_v=d_v,
                                   head=head,
                                   p=p)
        
        self.feed_forward = FeedForward(d_model=d_model,
                                        d_ff=d_ff,
                                        p=p)

    def forward(self, x: torch.Tensor, enc_mask: torch.Tensor):
        """forward 함수

        Args:
            x (torch.Tensor(bs, enc_len, d_model)): foward 입력값
            enc_mask (torch.Tensor(bs, 1, enc_len, enc_len)): encoder mask

        Returns:
            output (torch.Tensor(bs, enc_len, d_model)): forward 출력값
        """

        x = self.attention(x, x, x, enc_mask)

        output = self.feed_forward(x)

        return output