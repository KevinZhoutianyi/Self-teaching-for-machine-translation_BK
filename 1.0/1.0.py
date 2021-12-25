# %% [markdown]
# reload each module run each cell

# %%
# %%
import os
os.getcwd() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")

# %%
from T5 import *
from datasets import load_dataset
from transformers import T5Tokenizer
from MT_hyperparams import *
import torch.backends.cudnn as cudnn
from utils import *
from attention_params import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.autograd import Variable
from losses import *
from architect import *

# %%
dataset = load_dataset('opus_euconst','en-fr')
# print(dataset)
# print(dataset['train'][5])

# %%
# Setting the seeds
np.random.seed(seed_)
torch.cuda.set_device(0)
cudnn.benchmark = True
torch.manual_seed(seed_)
cudnn.enabled=True
torch.cuda.manual_seed(seed_)

# %%
# Load the tokenizer.
import random
tokenizer = T5Tokenizer.from_pretrained("t5-base")

criterion = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id, reduction='none')
L = len(dataset['train'])
L_t = L//4*3
L_v = L//8
L_test = L//8
dataset = dataset.shuffle(seed=seed_)



train = dataset['train']['translation'][:L_t]
valid = dataset['train']['translation'][L_t:L_t+L_v]
test = dataset['train']['translation'][-L_test:]

# %%
def preprocess(dat):
    for t in dat:
        t['en'] = 'translate English to French:' + t['en']
preprocess(train)
preprocess(valid)
preprocess(test)

# %%
# print("train len:",len(train))
# print("valid len:",len(valid))
# print("test len:" ,len(test))
# print(train[5])
type(train)

# %%
train_data = get_train_Dataset(train, tokenizer)# Create the DataLoader for our training set.
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=2, pin_memory=True, num_workers=0)

# %%
# load the attention parameters
A = attention_params(len(train))
# attention_weights.load_state_dict(torch.load(os.path.join(args.save, 'A.pt')))
A = A.cuda()

# %%
valid_data = get_aux_dataset(valid, tokenizer)# Create the DataLoader for our training set.
valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data), 
                        batch_size=2, pin_memory=True, num_workers=0)

# %%
test_data = get_aux_dataset(test, tokenizer)# Create the DataLoader for our training set.
test_dataloader = DataLoader(test_data, 
                        batch_size=5, pin_memory=True, num_workers=0)#, sampler=RandomSampler(test_data)

# %%

from MT_hyperparams import *

# %%
model_w = T5(criterion=criterion, tokenizer= tokenizer)
# model.load_state_dict(torch.load(os.path.join(args.save, 'gpt_weights.pt')))
model_w = model_w.cuda()
w_optimizer = torch.optim.SGD(model_w.parameters(),lr,momentum=momentum,weight_decay=decay)
scheduler_w  = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, float(epochs), eta_min=learning_rate_min)




# %%
model_v = T5(criterion=criterion, tokenizer= tokenizer)
# model.load_state_dict(torch.load(os.path.join(args.save, 'gpt_weights.pt')))
model_v = model_v.cuda()
v_optimizer = torch.optim.SGD(model_v.parameters(),lr,momentum=momentum,weight_decay=decay)
scheduler_v  = torch.optim.lr_scheduler.CosineAnnealingLR(v_optimizer, float(epochs), eta_min=learning_rate_min)

# %%
x = ['my name is kevin','it is my nameit is my nameit is my name 321312']
for index,i in enumerate(x) :
    x[index] = 'translate English to French:' + x[index]
y= tokenize(x, tokenizer, max_length = summary_length)
input = y[0].cuda()
output  = model_v.generate(input)
tokenizer.batch_decode(output)

# %%
def my_test(test_dataloader,model):
    for step, batch in enumerate(test_dataloader):
        x = Variable(batch[0], requires_grad=False).cuda()
        x_attn = Variable(batch[1], requires_grad=False).cuda()
        y = Variable(batch[2], requires_grad=False).cuda()
        y_attn = Variable(batch[3], requires_grad=False).cuda()

        ls = my_loss(x,x_attn,y,y_attn,model)
        # print('\n test loss :',ls)
        break
        

# %%
def my_train(epoch, train_dataloader, valid_dataloader, w_model, v_model, architect, A, w_optimizer, v_optimizer, lr_w, lr_v, ):
    for step, batch in enumerate(train_dataloader):
        # for index,t in enumerate(batch):
        #     # print("Training data ",index,"'s shape ",t.shape,end=' ')
        batch_loss_w, batch_loss_v,  batch_count = 0, 0, 0
        input_w = Variable(batch[0], requires_grad=False).cuda()
        input_w_attn = Variable(batch[1], requires_grad=False).cuda()
        output_w = Variable(batch[2], requires_grad=False).cuda()
        output_w_attn = Variable(batch[3], requires_grad=False).cuda()        
        input_v = Variable(batch[4], requires_grad=False).cuda()
        input_v_attn = Variable(batch[5], requires_grad=False).cuda()        
        # attention indices for CTG loss
        attn_idx = Variable(batch[6], requires_grad=False).cuda()
        
        #####################################################################################
        # valid 

        # valid input_valid, target_valid, valid_attn_classifier
        
        # get a random minibatch from the search queue with replacement
        valid_batch = next(iter(valid_dataloader))

        valid_input_v      = Variable(valid_batch[0], requires_grad=False).cuda()
        valid_input_v_attn = Variable(valid_batch[1], requires_grad=False).cuda()
        valid_out_v      = Variable(valid_batch[2], requires_grad=False).cuda()
        valid_out_v_attn = Variable(valid_batch[3], requires_grad=False).cuda()


        if begin_epoch <= epoch <= stop_epoch:
            
            architect.step(input_w,  output_w,input_w_attn, output_w_attn, w_optimizer, input_v, input_v_attn,valid_input_v, valid_input_v_attn, valid_out_v, 
                valid_out_v_attn, v_optimizer, attn_idx, lr_w, lr_v)
        # end the framework training and just train on the classifier task after the stop epoch
        if epoch <=stop_epoch:
            ######################################################################
            # Update the W model
            w_optimizer.zero_grad()
            

            # W
            loss_w = CTG_loss(input_w, input_w_attn, output_w, output_w_attn, attn_idx, A, w_model)
            # store the batch loss
            batch_loss_w += loss_w.item()

            loss_w.backward()
            
            nn.utils.clip_grad_norm(w_model.parameters(), grad_clip)
            
            w_optimizer.step()
            # # print(w_optimizer)
            
            ######################################################################
            # Update the V model
            v_optimizer.zero_grad()
        
            # the training loss
            # logits, loss_tr = w_model.loss(article_DS, article_DS_attn, summary_DS, summary_DS_attn)

            # Loss on augmented dataset
            
            loss_aug = calc_loss_aug(input_v, input_v_attn, w_model, v_model)
        
            v_loss =  (loss_aug)
            batch_loss_v += v_loss.item()
            
            v_loss.backward()
            
            nn.utils.clip_grad_norm(v_model.parameters(), grad_clip)
            
            # update the classifier model
            v_optimizer.step()     
            
            my_test(test_dataloader,w_model) 
            my_test(test_dataloader,v_model)      


# %%

architect = Architect(model_w, model_v,  A)

# %%
# a = [[13959,  1566,    12,  2379,    10, 17608,   994,    27,     1,     0],[13959,  1566,    12,  2379,    10, 17608,   994,    27,     1,     0]]
# aa = torch.LongTensor(a)
# aa.long()
# b=torch.zeros(aa.shape[0],aa.shape[1],32128)
# c = b.scatter_(-1,aa.unsqueeze(-1), 1.).float().cuda()
# model_v(c,torch.ones_like(c))

# %%
my_train(begin_epoch, train_dataloader, valid_dataloader, model_w, model_v,  architect, A, w_optimizer, v_optimizer, lr,lr)
    

# %%
import gc

gc.collect()

torch.cuda.empty_cache()

# %%
tokenizer.decode([0,  6206,  6667,    27,     1])
tokenizer.decode([13959,  1566,    12,  2379,    10, 17608,   994,    27,     1,     0])
# print(model_v.vocab_size)
logit = torch.load('logits.pt')
target = torch.load('target_ids.pt')
tokenizer.decode(target[0])
logit.shape
_,maxx = torch.max(logit,dim=-1,keepdim=True)
maxx.shape
tokenizer.decode(maxx[0].squeeze(-1))


# %%
model_v.embedding

# %%



