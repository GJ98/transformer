import torch
from torch import nn

from transformer.feed_forwards.position_wise_feed_forward import PositionWiseFeedForward


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, p: int):
        """feed forward sub layer 구현 클래스

        Args:
            d_model (int):  intput, output dim
            d_ff (int): hidden dim
            p (int): dropout probability
        """

        super().__init__()
        self.feed_forward = PositionWiseFeedForward(d_model=d_model,
                                                    d_ff=d_ff,
                                                    p=p)

        self.dropout = nn.Dropout(p=p)

        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        """forward 함수

        Args:
            x (torch.Tensor(bs, len, d_model)): forward 입력값

        Returns:
            output (torch.Tensor(bs, len, d_model)): forward 출력값
        """

        residual = x

        x = self.feed_forward(x)

        x = self.norm(x + residual)

        output = self.dropout(x)

        return output

