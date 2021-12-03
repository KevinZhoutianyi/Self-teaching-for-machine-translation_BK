import os
import gc
from losses import *
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from MT_hyperparams import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):

    def __init__(self, w_model, v_model, DS_model, A):

        self.w_momentum = momentum
        self.w_decay = decay

        self.v_momentum =momentum
        self.v = decay

        self.w_model = w_model

        self.v_model = v_model


        self.A = A

        # change to ctg dataset importance 
        # change to .parameters()

        self.optimizer_A = torch.optim.Adam(self.A.parameters(), 
          lr=lr, betas=(0.5, 0.999), weight_decay=decay)



    #########################################################################################
    # Computation of G' model named as unrolled model
    
    def _compute_unrolled_w_model(self, input, target, input_attn, target_attn, attn_idx, eta_w, w_optimizer):
        # BART loss
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.A, self.w_model)
        # Unrolled model
        theta = _concat(self.w_model.parameters()).data
        dtheta = _concat(torch.autograd.grad(loss, self.w_model.parameters(), retain_graph = True )).data + self.w_decay*theta
        
        # convert to the model
        unrolled_w_model = self._construct_w_model_from_theta(theta.sub(eta_w, dtheta))
        return unrolled_w_model

    # reshape the w model parameters
    def _construct_w_model_from_theta(self, theta):
        
        model_dict = self.w_model.state_dict()
    
        # create the new bart model
        w_model_new = self.w_model.new()

        # encoder update
        params, offset = {}, 0
        for k, v in self.w_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        w_model_new.load_state_dict(model_dict)

        return w_model_new

    # update the bart model with one step gradient update for unrolled model

    #########################################################################################
    
    #####################################################################################################
    # Computation of 'DS' model named as unrolled model
    
    def _compute_unrolled_v_model(self, input, input_attn,  unrolled_w_model,  eta_v, v_optimizer):

        # DS loss on augmented dataset
        loss_aug = calc_loss_aug(input, input_attn, unrolled_w_model,self.v_model)

        loss = (self.lambda_par*loss_aug)
        # Unrolled model
        theta = _concat(self.v_model.parameters()).data
        try:
            moment = _concat(v_optimizer.state[v]['momentum_buffer'] for v in self.v_model.parameters()).mul_(self.DS_network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.v_model.parameters(), retain_graph = True )).data + self.DS_network_weight_decay*theta
        
        # convert to the model
        unrolled_v_model = self._construct_v_model_from_theta(theta.sub(eta_v, moment+dtheta))

        return unrolled_v_model

    # reshape the T model parameters
    def _construct_v_model_from_theta(self, theta):
        
        model_dict = self.v_model.state_dict()
    
        # create the new bart model
        v_model_new = self.v_model.new()

        # encoder update
        params, offset = {}, 0
        for k, v in self.v_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        v_model_new.load_state_dict(model_dict)

        return v_model_new

    #########################################################################################

    # one step update for the importance parameter A
    def step(self, article_DS, article_DS_attn, summary_DS, summary_DS_attn, article_bart, article_bart_attn, summary_bart, 
        summary_bart_attn, summary_gpt, summary_gpt_attn, valid_article_DS, valid_summary_DS, valid_article_DS_attn, 
        valid_summary_DS_attn, attn_idx, eta_gpt, eta_bart, eta_DS, gpt_optimizer, bart_optimizer, DS_optimizer):
        
        # self.optimizer_A.zero_grad()

        # self.optimizer_B.zero_grad()

        # unrolled_gpt_model = self._compute_unrolled_gpt_model(summary_gpt, summary_gpt, summary_gpt_attn, summary_gpt_attn, attn_idx, eta_gpt, gpt_optimizer)

        # unrolled_gpt_model.gpt_model.train()

        # unrolled_bart_model = self._compute_unrolled_bart_model(summary_bart, article_bart, summary_bart_attn, article_bart_attn, attn_idx, eta_bart, bart_optimizer)

        # unrolled_bart_model.bart_model.train()

        # unrolled_DS_model = self._compute_unrolled_DS_model(article_DS, summary_DS, summary_gpt, article_DS_attn, summary_DS_attn, summary_gpt_attn, unrolled_gpt_model, unrolled_bart_model, attn_idx, eta_DS, DS_optimizer)

        # _, unrolled_DS_loss = unrolled_DS_model.loss(valid_article_DS, valid_article_DS_attn, valid_summary_DS, valid_summary_DS_attn)

        # unrolled_DS_model.train()

        # unrolled_DS_loss.backward()

        # vector_s_dash = [v.grad.data for v in unrolled_DS_model.parameters()]

        # implicit_grads_B = self._outer_B(vector_s_dash, article_bart, article_bart_attn, summary_bart, summary_bart_attn, summary_gpt, summary_gpt_attn, 
        #     attn_idx, unrolled_gpt_model, unrolled_bart_model, eta_bart, eta_DS)

        # implicit_grads_A = self._outer_A(vector_s_dash, summary_gpt, summary_gpt_attn, attn_idx, unrolled_gpt_model, 
        #     unrolled_bart_model, eta_gpt, eta_DS)

        # # change to ctg dataset importance
        # # change to .parameters()
        # for v, g in zip(self.B.parameters(), implicit_grads_B):
        #     if v.grad is None:
        #         v.grad = Variable(g.data)
        #     else:
        #         v.grad.data.copy_(g.data)

        # self.optimizer_B.step()

        # # change to ctg dataset importance
        # # change to .parameters()
        # for v, g in zip(self.A.parameters(), implicit_grads_A):
        #     if v.grad is None:
        #         v.grad = Variable(g.data)
        #     else:
        #         v.grad.data.copy_(g.data)

        # self.optimizer_A.step()

        # del unrolled_gpt_model

        # del unrolled_bart_model

        # del unrolled_DS_model

        # gc.collect()

    ######################################################################
    # finite difference approximation of the hessian and the vector product for T
    def _hessian_vector_product_B(self, vector, input, target, input_attn, target_attn, attn_idx, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.bart_model.bart_model.parameters(), vector):
            p.data.add_(R, v)
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.B, self.bart_model)

        # change to ctg dataset importance
        grads_p = torch.autograd.grad(loss, self.B.parameters())

        for p, v in zip(self.bart_model.bart_model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.B, self.bart_model)

        # change to ctg dataset importance
        # change to .parameters()
        grads_n = torch.autograd.grad(loss, self.B.parameters())

        for p, v in zip(self.bart_model.bart_model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


    ######################################################################
    # function for the product of hessians and the vector product wrt T and function for the product of
    # hessians and the vector product wrt G
    def _outer_B(self, vector_s_dash, article_bart, article_bart_attn, summary_bart, summary_bart_attn, summary_gpt, summary_gpt_attn, 
        attn_idx, unrolled_gpt_model, unrolled_bart_model, eta_bart, eta_DS, r=1e-2):
        
        R1 = r / _concat(vector_s_dash).norm()

        # plus S
        for p, v in zip(self.DS_model.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        unrolled_bart_model.bart_model.train()

        # use the bart model

        loss_aug_p = calc_loss_aug(summary_gpt, summary_gpt_attn, unrolled_gpt_model, unrolled_bart_model, self.DS_model)

        # T
        vector_dash = torch.autograd.grad(loss_aug_p, unrolled_bart_model.bart_model.parameters(), retain_graph = True)

        grad_part1 = self._hessian_vector_product_B(vector_dash, summary_bart, article_bart, summary_bart_attn, article_bart_attn, attn_idx)

        # minus S
        for p, v in zip(self.DS_model.parameters(), vector_s_dash):
            p.data.sub_(2*R1, v)

        loss_aug_m = calc_loss_aug(summary_gpt, summary_gpt_attn, unrolled_gpt_model, unrolled_bart_model, self.DS_model)
        
        # T

        vector_dash = torch.autograd.grad(loss_aug_m, unrolled_bart_model.bart_model.parameters(), retain_graph = True)

        grad_part2 = self._hessian_vector_product_B(vector_dash, summary_bart, article_bart, summary_bart_attn, article_bart_attn, attn_idx)

        for p, v in zip(self.DS_model.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        grad = [(x-y).div_((2*R1)/(eta_bart*eta_DS*self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]

        return grad


    ######################################################################
    # finite difference approximation of the hessian and the vector product for T
    def _hessian_vector_product_A(self, vector, input, target, input_attn, target_attn, attn_idx, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.gpt_model.gpt_model.parameters(), vector):
            p.data.add_(R, v)
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.A, self.gpt_model)

        # change to ctg dataset importance
        grads_p = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.gpt_model.gpt_model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.A, self.gpt_model)

        # change to ctg dataset importance
        # change to .parameters()
        grads_n = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.gpt_model.gpt_model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


    ######################################################################
    # function for the product of hessians and the vector product wrt T and function for the product of
    # hessians and the vector product wrt G
    def _outer_A(self, vector_s_dash, summary_gpt, summary_gpt_attn, attn_idx, unrolled_gpt_model, 
        unrolled_bart_model, eta_gpt, eta_DS, r=1e-2):
        
        R1 = r / _concat(vector_s_dash).norm()

        # plus S
        for p, v in zip(self.DS_model.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        unrolled_gpt_model.gpt_model.train()

        # use the gpt model

        loss_aug_p = calc_loss_aug(summary_gpt, summary_gpt_attn, unrolled_gpt_model, unrolled_bart_model, self.DS_model)

        # T
        vector_dash = torch.autograd.grad(loss_aug_p, unrolled_gpt_model.gpt_model.parameters(), retain_graph = True)

        grad_part1 = self._hessian_vector_product_A(vector_dash, summary_gpt, summary_gpt, summary_gpt_attn, summary_gpt_attn, attn_idx)

        # minus S
        for p, v in zip(self.DS_model.parameters(), vector_s_dash):
            p.data.sub_(2*R1, v)

        loss_aug_m = calc_loss_aug(summary_gpt, summary_gpt_attn, unrolled_gpt_model, unrolled_bart_model, self.DS_model)
        
        # T

        vector_dash = torch.autograd.grad(loss_aug_m, unrolled_gpt_model.gpt_model.parameters(), retain_graph = True)

        grad_part2 = self._hessian_vector_product_A(vector_dash, summary_gpt, summary_gpt, summary_gpt_attn, summary_gpt_attn, attn_idx)

        for p, v in zip(self.DS_model.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        grad = [(x-y).div_((2*R1)/(eta_gpt*eta_DS*self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]

        return grad

# print("123")