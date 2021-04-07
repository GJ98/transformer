import torch
from torch import nn

from transformer.embeddings.positional_encoding import PositionalEncoding


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, len: int, d_model: int, device):
        """embedding 구현 클래스

        Args:
            vocab_size (int): vocabulary size
            len (int): length
            d_model (int): embedding dim
            device : device type
        """

        super().__init__()
        # pad
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=d_model)

        self.pos_enc = PositionalEncoding(len=len,
                                          d_model=d_model,
                                          device=device)
                        
    def forward(self, x: torch.Tensor):
        """forward 함수

        Args:
            x (torch.Tensor(bs, len)): embedding 입력값

        Returns:
            output (torch.Tensor(bs, len, d_model)): embedding 결과값
        """

        embed = self.embed(x)

        pos_enc = self.pos_enc(x)

        output = embed + pos_enc

        return output