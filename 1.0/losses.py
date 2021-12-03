import os
import gc
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from hyperparams import *

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
    
    # similar to the loss defined in the BART model hugging face conditional text generation
    
    # probability predictions
    loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
    
    loss = torch.dot(attention_weights, loss_vec)
    
    scaling_factor = 1
    
    return scaling_factor*loss


# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug

def calc_loss_aug(input_ids, input_attn, w_model, v_model):

    #######################################################################################################
    ### GPT convert
    # convert input to the bart encodings
    # # use the generate approach
    output_ids = w_model.generate(input_ids[:,:15])
    
    gpt_logits = w_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

    # find the decoded vector from probabilities
    
    gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

    #######################################################################################################
    
    # convert to the bart ids
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids), bart_model.tokenizer, max_length = summary_length)
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

    #######################################################################################################

    ## BART model articles generation
    
    # the gumbel soft max trick
    
    one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
    bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

    # BART
    bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
    bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

    # find the decoded vector from probabilities
    
    bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True)

    #######################################################################################################

    # convert to the DS ids

    bart2DS_article_ids, bart2DS_article_attn = tokenize(bart_model.tokenizer.batch_decode(bart_article_ids), DS_model.tokenizer, max_length = article_length)
    
    bart2DS_article_ids, bart2DS_article_attn = bart2DS_article_ids.cuda(), bart2DS_article_attn.cuda()
    
    bart2DS_summary_ids, bart2DS_summary_attn = tokenize(bart_model.tokenizer.batch_decode(gpt2bart_summary_ids), DS_model.tokenizer, max_length = summary_length)
    
    bart2DS_summary_ids, bart2DS_summary_attn = bart2DS_summary_ids.cuda(), bart2DS_summary_attn.cuda()

    #######################################################################################################
    
    # DS model

    one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.vocab_size).cuda()
    
    DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    
    
    DS_summary_ids = bart2DS_summary_ids
    
    loss = DS_model(DS_article_ids, bart2DS_article_attn, target_ids = DS_summary_ids, target_attn = bart2DS_summary_attn).loss
        
    del DS_article_ids, bart_summary_ids, one_hot
        
    gc.collect()   
    
    return loss