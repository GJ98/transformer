import torch
from torch import nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        """position wise feed forward 구현 클래스
        
        Args:
            d_model (int): input, output dim
            d_ff (int): hidden dim
        """

        super().__init__()
        self.w_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff, out_features=d_model)

        self.relu = nn.ReLU()

    def forward(self, x):
        """forward 함수

        Args:
            x (torch.Tensor(bs, len, d_model)): forward 입력값
        
        Returns:
            output (torch.Tensor(bs, len, d_model)): forward 결과값
        """

        x = self.w_1(x)

        x = self.relu(x)

        output = self.w_2(x)

        return output
