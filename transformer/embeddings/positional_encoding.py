import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, len: int, d_model: int, device):
        """positional encoding 구현 클래스
        
        Args:
            len (int): length
            d_model (int): embedding dim
            device : device type
        """

        super().__init__()
        self.pos_enc = torch.zeros(len, d_model, device=device)
        self.pos_enc.requires_grad = False

        pos = torch.arange(start=0, end=len, device=device).unsqueeze(1)

        _2i = torch.arange(start=0, end=d_model, step=2, device=device)

        self.pos_enc[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.pos_enc[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x: torch.Tensor):
        """forward 함수

        Args:
            x (torch.Tensor(bs, len)): positional encoding 입력값

        Returns:
            output(torch.Tensor(len, d_model)): position encoding 결과값
        """

        bs, len = x.size()

        return self.pos_enc[:len, :]

        


        