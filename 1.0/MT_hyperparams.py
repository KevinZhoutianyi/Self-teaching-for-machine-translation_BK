import numpy as np
import torch

seed_ = 1024

summary_length = 100

article_length = 500

epochs = 25
ux_ratio = 0.3
learning_rate_min = 0.01
lr = 0.01
begin_epoch = 1
stop_epoch = 5
momentum = 0.1
grad_clip =  5
decay = 0