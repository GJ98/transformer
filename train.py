import os
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *
from data import *
from metrics import scoring

from transformer.transformer import Transformer

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
                    device=device)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

optimizer = Adam(params=model.parameters(),
                 lr=model_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                 factor=lr_scheduler_factor,
                                 min_lr=lr_scheduler_min_lr,
                                 patience=lr_scheduler_patience)

for epoch in tqdm(range(epoch), desc='epoch', total=epoch):
    tqdm.write("\nepoch : {}, lr : {}".format(epoch,
                                              optimizer.param_groups[0]['lr']))

    tr_loss, tr_acc, total_num = 0, 0, 0
    model.to(device)
    model.train()
    print('train')

    for data in tqdm(train_iter, desc='step', total=len(train_iter)):
        optimizer.zero_grad()

        source = data.src.T.contiguous()
        target = data.trg.T.contiguous()

        output = model(source, target[:, :-1])

        output = output.reshape(-1, output.size(-1))
        target = target[:, 1:].contiguous().view(-1)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)

        with torch.no_grad():
            _correct_num, _total_num = scoring(output, target)
            tr_loss += loss.item()
            tr_acc += _correct_num
            total_num += _total_num

    tr_loss_avg = tr_loss / len(train_iter)
    tr_acc_avg = tr_acc / total_num

    tqdm.write("\nepoch: {}, tr_loss: {}, tr_acc: {}".format(epoch,
                                                             tr_loss_avg,
                                                             tr_acc_avg))

    if epoch % 5 == 0:
        state_dict = model.to(torch.device('cpu')).state_dict()

        torch.save(state_dict, model_file)