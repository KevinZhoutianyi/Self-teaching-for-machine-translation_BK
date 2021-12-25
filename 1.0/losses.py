import os
import gc
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

# the loss for the encoder and the decoder model 
# this takes into account the attention for all the datapoints for the encoder-decoder model
def CTG_loss(input_ids, input_attn, target_ids, target_attn, attn_idx, attention_parameters, model):
    
    attention_weights = attention_parameters(attn_idx)
    # print(attention_weights)
    # similar to the loss defined in the BART model hugging face conditional text generation
    
    # probability predictions
    # print("CTG loss shape,",input_ids.shape, input_attn.shape,target_ids.shape, target_attn.shape)
    loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
    
    loss = torch.dot(attention_weights, loss_vec)
    
    scaling_factor = 1
    
    return scaling_factor*loss


# normal loss
def my_loss(input_ids, input_attn, target_ids, target_attn, model):
    
    
    # print("CTG loss shape,",input_ids.shape, input_attn.shape,target_ids.shape, target_attn.shape)
    loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
    loss = torch.sum(loss_vec,dim=0)
    
    return loss


# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug

def calc_loss_aug(input_ids, input_attn, w_model, v_model):

    print("input_ids",input_ids.shape)
    output_ids = w_model.generate(input_ids)
    print("output_ids",output_ids.shape)
    # print("output_ids",output_ids)
    w_logits = w_model(input_ids, input_attn, target_ids = output_ids, target_attn = torch.ones_like(output_ids).long()).logits
    print("w_logits",w_logits.shape)
    w_soft_idx, bart_idx = torch.max(w_logits, dim=-1, keepdims= True)

    
    print("w_soft_idx",w_soft_idx.shape)
    one_hot = torch.zeros(input_ids.shape[0], input_ids.shape[1], v_model.vocab_size).cuda()
    print("one_hot",one_hot.shape)
    w_output_ids = one_hot.scatter_(-1, output_ids.unsqueeze(-1), 1.).float().detach() + w_soft_idx.sum() - w_soft_idx.sum().detach()
    print("w_output_ids",w_output_ids.shape)
    # w_output_ids = w_output_ids.long()
    loss = v_model.loss( w_output_ids,  torch.ones_like(input_ids).long(), target_ids = input_ids, target_attn = input_attn)
    #here  w_output_ids is a catagory distribution, input_ids are index
    loss = torch.sum(loss,dim=0)
    # print("loss",loss.shape)
    return loss