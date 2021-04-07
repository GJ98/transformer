import torch
from torch import nn

from transformer.layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, head: int, d_ff: int, n_layer: int, p: int):
        """encoder 구현 클래스

        Args:

            d_model (int): intput, output dim
            d_k (int): key, query dim
            d_v (int): value dim
            head (int): parallel attention layers
            d_ff (int): hidden dim
            layer_num (int) = number of layer
            p (int): dropout probability
        """

        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, 
                                                  d_k=d_k, 
                                                  d_v=d_v, 
                                                  head=head, 
                                                  d_ff=d_ff,
                                                  p=p) for _ in range(n_layer)])

    def forward(self, x: torch.Tensor, enc_mask: torch.Tensor):
        """forward 함수

        Args:
            x (torch.Tensor(bs, enc_len, d_model)): foward 입력값
            enc_mask(torch.Tensor(bs, 1, enc_len, enc_len)): encoder mask

        Returns:
            output (torch.Tensor(bs, enc_len, d_model)): forward 출력값
        """ 

        for layer in self.layers:
            x = layer(x, enc_mask)

        output = x
        
        return output

