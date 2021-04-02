import torch
from torch import nn

from transformer.sub_layers.attention import Attention
from transformer.sub_layers.feed_forward import FeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, head: int, d_ff: int):
        """decoder layer 구현 클래스

        Args:
            d_model (int): intput, output dim
            d_k (int): key, query dim
            d_v (int): value dim
            head (int): parallel attention layers
            d_ff (int): hidden dim
        """

        super().__init__()
        self.attention = Attention(d_model=d_model,
                                   d_k=d_k,
                                   d_v=d_v,
                                   head=head)
        
        self.feed_forward = FeedForward(d_model=d_model,
                                        d_ff=d_ff)

    def forward(self, 
                enc_out: torch.Tensor, 
                dec_in: torch.Tensor, 
                dec_mask: torch.Tensor, 
                enc_dec_mask: torch.Tensor):
        """forward 함수

        Args:
            enc_out (torch.Tensor(bs, enc_len, d_model)): encoder 출력값
            dec_in (torch.Tensor(bs, dec_len, d_model)): previous decoder layer 입력값
            dec_mask (torch.Tensor(bs, dec_len, dec_len)): decoder mask
            enc_dec_mask (torch.Tensor(bs, dec_len, enc_len)): encoder decoder mask

        Returns:
            output (torch.Tensor(bs, dec_len, d_model)): forward 출력값
        """

        x = self.attention(dec_in, dec_in, dec_in, dec_mask)

        x = self.attention(enc_out, enc_out, x, enc_dec_mask)

        output = self.feed_forward(x)

        return output