import os
import platform
import torch

#device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#path
root_dir = os.path.abspath(os.curdir)
_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _
model_dir = root_dir + "saved{_}".format(_=_)
model_file = model_dir + "transformer"

#model
batch_size=64
d_model = 512
d_k = 64
d_v = 64
head = 8
d_ff = 2048
n_layer = 6
enc_len = 256
dec_len = 256
p = 0.1

#optimizer
epoch = 1000
model_lr = 0.1
adam_eps = 5e-9
weight_decay = 5e-4
lr_scheduler_factor = 0.9
lr_scheduler_min_lr = 1e-10
lr_scheduler_patience = 10
