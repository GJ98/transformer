import torch
from torch import nn

from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder
from transformer.embeddings.embedding import Embedding


class Transformer(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 d_k: int, 
                 d_v: int, 
                 head: int, 
                 d_ff: int, 
                 n_layer: int, 
                 enc_vocab_size: int,
                 dec_vocab_size: int,
                 enc_len: int,
                 dec_len: int,
                 pad: int,
                 device):
        """transformer 구현 클래스

        Args:
            d_model (int): intput, output dim
            d_k (int): key, query dim
            d_v (int): value dim
            head (int): parallel attention layers
            d_ff (int): hidden dim
            n_layer (int): number of layer
            enc_vocab_size (int): decoder vocab size
            dec_vocab_size (int): decoder vocab size
            enc_len (int): encoder length
            dec_len (int): decoder length
            pad (int): pad idx
            device : device type
        """

        super().__init__()
        self.pad = pad
        self.device = device
        self.enc_embed = Embedding(vocab_size=enc_vocab_size,
                                   len=enc_len,
                                   d_model=d_model,
                                   device=device)

        self.dec_embed = Embedding(vocab_size=dec_vocab_size,
                                   len=dec_len,
                                   d_model=d_model,
                                   device=device)

        self.encoder = Encoder(d_model=d_model,
                               d_k=d_k,
                               d_v=d_v,
                               head=head,
                               d_ff=d_ff,
                               n_layer=n_layer)

        self.decoder = Decoder(d_model=d_model,
                               d_k=d_k,
                               d_v=d_v,
                               head=head,
                               d_ff=d_ff,
                               n_layer=n_layer)
                            
        self.linear = nn.Linear(in_features=d_model,
                                out_features=dec_vocab_size)

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor):
        """forward 함수

        Args:
            enc_in (torch.Tensor(bs, enc_len)): encoder 입력값
            dec_in (torch.Tensor(bs, dec_len)): decoder 입력값

        Returns:
            output (torch.Tensor(bs, dec_len, vocab_size)): transformer 출력값
        """

        enc_mask = self.get_pad_mask(enc_in, enc_in, self.pad)
        
        dec_pad_mask = self.get_pad_mask(enc_in, enc_in, self.pad)
        dec_no_peak_mask = self.get_no_peak_mask(enc_in, enc_in)
        dec_mask = dec_pad_mask * dec_no_peak_mask

        enc_dec_mask = self.get_pad_mask(dec_in, enc_in, self.pad)

        enc_in = self.enc_embed(enc_in)
        dec_in = self.dec_embed(dec_in)

        enc_out = self.encoder(enc_in, enc_mask) 
        dec_out = self.decoder(enc_out, dec_in, dec_mask, enc_dec_mask)
        output = self.linear(dec_out)

        return output

    def get_pad_mask(self, q: torch.Tensor, k: torch.Tensor, pad: int):
        """pad mask 수행 함수

        Args:
            q (torch.Tensor(bs, len_q)): query
            k (torch.Tensor(bs, len_k)): key
            pad (int): pad id

        Returns:
            mask (torch.Tensor(bs, len_q, len_k)): pad mask
        """

        len_q = q.size(1)

        mask = k.ne(pad).unsqueeze(1).repeat(1, len_q, 1).float()

        return mask

    def get_no_peak_mask(self, q: torch.Tensor, k: torch.Tensor):
        """sub mask 수행 함수

        Args:
            q (torch.Tensor(bs, len_q)): query
            k (torch.Tensor(bs, len_k)): key

        Returns:
            mask (torch.Tensor(bs, len_q, len_k)): no peak mask
        """

        bs, len_q, len_k = q.size(0), q.size(1), k.size(1)

        mask = torch.tril(torch.ones(bs, len_q, len_k)).to(self.device)

        return mask