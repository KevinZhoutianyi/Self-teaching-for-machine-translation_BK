import numpy as np
import torch

seed_ = 2

summary_length = 100

article_length = 500

epochs = 25
ux_ratio = 0.3
learning_rate_min = 0.001
w_lr = 10
v_lr = 0.01
A_lr = 1
begin_epoch = 1
stop_epoch = 5
momentum = 0
grad_clip =  5
decay = 0