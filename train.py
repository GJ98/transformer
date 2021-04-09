import os
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *
from data import *
from bleu import *

from transformer.transformer import Transformer

"""
ref : https://tutorials.pytorch.kr/beginner/torchtext_translation_tutorial.html
"""

model = Transformer(d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    head=head,
                    d_ff=d_ff,
                    n_layer=n_layer,
                    enc_vocab_size=enc_voc_size,
                    dec_vocab_size=dec_voc_size,
                    enc_len=enc_len,
                    dec_len=dec_len,
                    pad=pad_idx,
                    p=p,
                    device=device).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                 verbose=True,
                                 factor=factor,
                                 patience=patience)

def train(model, iterator, optimizer, loss_fn):
    model.train()
    epoch_loss = 0
    for batch in tqdm(iterator, desc='step', total=len(iterator)):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output = output.contiguous().reshape(-1, output.size(-1))
        trg = trg[:, 1:].contiguous().view(-1)

        loss = loss_fn(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, loss_fn):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for batch in tqdm(iterator, desc='step', total=len(iterator)):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.reshape(-1, output.size(-1))
            trg_reshape = trg[:, 1:].contiguous().view(-1)

            loss = loss_fn(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            # bleu score calculation
            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(trg[j], trg_vocab)
                    output_words = idx_to_word(output[j].max(dim=1)[1], trg_vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for epoch in range(total_epoch):
        train_loss = train(model, train_iter, optimizer, loss_fn)
        valid_loss, bleu = evaluate(model, valid_iter, loss_fn)

        print(bleu)
        if epoch > warmup:
            lr_scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        if valid_loss < best_loss and epoch % 10 == 0:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_dir + 'model-{}.pt'.format(valid_loss))

        f = open(result_dir + 'train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open(result_dir + 'bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open(result_dir + 'test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print('epoch : {} \t train loss : {:.3f} \t val loss : {:.3f} \t bleu : {:.3f}'.format(epoch + 1,
                                                                                               train_loss,
                                                                                               valid_loss,
                                                                                               bleu))

if __name__ == '__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    run(total_epoch=epoch, best_loss=inf)