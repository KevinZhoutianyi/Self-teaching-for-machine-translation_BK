import os
import random
import numpy as np
import copy
from transformers import  T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from MT_hyperparams import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)


class Embedding_(torch.nn.Module):
    def __init__(self, embedding_layer):
        super(Embedding_, self).__init__()
        self.embedding = embedding_layer.cuda()

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.embedding.weight)


class T5(nn.Module):
    
    def __init__(self, criterion, tokenizer, MODEL = 't5-base'):
        super(T5, self).__init__()


        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        
        self._criterion = criterion

        self.model = T5ForConditionalGeneration.from_pretrained(MODEL)
        
        # print('Loading the pretrained model ....')
        # Load the pre-trained model trained for 
        # self.model.load_state_dict(torch.load('pretrained_BART.pt'))
        # print('Done! Loaded the pretrained model ....')
        
        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()

        # embedding layer for both encoder and decoder since it is shared   
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()#convert token to 512dimensions vector
        self.enc_emb_scale = 1

    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):

        inp_emb = self.embedding(input_ids)/self.enc_emb_scale
        # print("T5 inputshape:",inp_emb.shape,input_attn.shape) # after embedding the shape becomes([5, 232, 768]) (batchsize,tokenziedlength,embeddinglength)
        out = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)

        return out
    
    def loss(self, input_ids, input_attn, target_ids, target_attn):

        output = self(input_ids, input_attn, target_ids, target_attn)

        return output.logits, output.loss

    def get_loss_vec(self, input_ids, input_attn, target_ids = None, target_attn = None):

        # batch size
        batch_size = target_ids.shape[0]
        
        # target_sequence_length of the model
        target_sequence_length = target_ids.shape[1]
        # print("getlossvec,loss input",input_ids.shape,target_ids.shape)
        logits = (self(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)).logits
        # print("getlossvec,logits",logits.shape)
        loss_vec = self._criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        # print("getlossvec,loss_vec",loss_vec.shape)
        loss_vec = loss_vec.view(batch_size, -1).mean(dim = 1)

        # print("getlossvec,loss_vec",loss_vec.shape)
        return loss_vec

    # used for generation of summaries from articles
    def generate(self, input_ids, num_beams = 2, max_length=article_length):
        
        # beam search
        output_ids = self.model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, no_repeat_ngram_size = 2, repetition_penalty = 1.2)
        
        ## sampling with top_p
        #summary_ids = self.model.generate( input_ids = input_ids, num_beams = 1, max_length = max_length, top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2, repetition_penalty = 1.2)

        return output_ids

    # new model for the definitions of gradients in architec.py 
    def new(self):

        # there is embedding layer and the summarization head that we will not train on 
        # we just train on the encoder and the decoder weights 
        model_new = T5(self._criterion, self.tokenizer).cuda()
        
        # hence we deep copy all the weights and update the required ones
        # use the first order approximation for the summarization head
        # i.e, do not use unrolled model w.r.t to the summarization head parameters
        model_new.model.load_state_dict(self.model.state_dict())
        
        return model_new


if __name__ == "__main__":
    print("T5 main")
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_criterion = torch.nn.CrossEntropyLoss(ignore_index = T5Tokenizer.pad_token_id, reduction='none')
    t5_criterion = t5_criterion.cuda()
    t5 = T5(t5_criterion,t5_tokenizer)
    t5 = t5.cuda()
    print(t5)

    

    t5new  = t5.new()
    print(t5new)