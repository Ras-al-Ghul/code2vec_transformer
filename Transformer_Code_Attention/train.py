'''
Reads the input dataset dump from a txt file and outputs learned
embedding weights pickles every x=3 epochs
'''

import re
import sys
import os
import time
import math
import json
import random
import argparse
import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from model import Model, LMHead
from opt import OpenAIAdam
from utils import (iter_data, ResultLogger, make_path)


class LossCompute:
    "A Loss compute and train function."
    def __init__(self, lm_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, M, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, 1:, 0].contiguous().view(-1)           # Shape: 252
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(1)-1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        
        # for validation
        if only_return_losses:
            return lm_losses

        # sum up the losses and backprop
        train_loss = lm_losses.sum()
        print(train_loss,)
        train_loss.backward()

        # using the optimizer
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

def transform_code(X):
    # set up positional embeddings
    # have to create bitmasks for positions with code tokens
    # which will be used while calculating the loss
    # as you only want to take those tokens for backpropping loss
    n_batch = len(X)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    for i, x, in enumerate(X):
        l = min(n_ctx, len(x)) #len(x)
        xmb[i, :l, 0] = x[:l]
        mmb[i, :l] = 1
    xmb[:, :, 1] = np.arange(n_vocab, n_vocab+n_ctx)
    return xmb, mmb

def iter_apply(Xs, Ms):
    # for validation loss
    # basically same process as run_epoch
    # except that you do a model.eval() to put the model in evaluation mode
    cost = []
    count = 0
    with torch.no_grad():
        model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            h = model(XMB)
            lm_logits = lm_head(h)
            lm_losses = compute_loss_fct(XMB, MMB, lm_logits, only_return_losses=True)
            print('val_iter', count, lm_losses.sum().item())
            cost.append(lm_losses.sum().item())
            count += 1
    return cost

def log(trV):
    # validation loss, does not backprop the losses computed
    print("Logging")
    if len(trV) % n_batch_train != 0:
        trV = trV[:len(trV) - (len(trV) % n_batch_train)]
    _trX, _trM = transform_code(trV)
    tr_cost = iter_apply(_trX[:], _trM[:])
    tr_cost = sum(tr_cost)/(len(_trM[:])/n_batch_train)
    print('tr_cost %.3f'%(tr_cost))

def get_train_valid(tr_input):
    # splits the data into train and valid
    n_train = len(tr_input)
    vp = int((valid_percent / 100) * n_train)
    start_ind = int(np.random.randint(0, n_train-vp))
    tr_valid = tr_input[start_ind:start_ind+vp]
    tr_train = tr_input[:start_ind] + tr_input[start_ind+vp:]
    return tr_train, tr_valid

def run_epoch(trX, trM):
    # training
    for xmb, mmb in iter_data(*shuffle(trX, trM, random_state=np.random),
                              n_batch=n_batch_train, truncate=True, verbose=True):
        # set the model in training mode
        model.train()
        # move the data to gpu
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        # pass the data through the model
        h = model(XMB)
        # pass the data through the lm head
        lm_logits = lm_head(h)
        # compute and backprop the loss
        compute_loss_fct(XMB, MMB, lm_logits)

argmax = lambda x:np.argmax(x, 1)

if __name__ == '__main__':
    # the model is configurable here
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='code')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=60)
    parser.add_argument('--n_batch', type=int, default=500)
    parser.add_argument('--n_batch_size', type=int, default=2000)    # num of lines of code in a batch
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=22)
    parser.add_argument('--n_embd', type=int, default=132)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=9)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)#4) # TODO add mutli-gpu training logic
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--src_folder', type=str, default='/jdk7u-jdk/')
    parser.add_argument('--train_file', type=str, default='./lex_dumps_data.txt')
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--valid_percent', type=float, default=15)

    args = parser.parse_args()
    pprint.pprint(args)
    globals().update(args.__dict__) #TODO maybe we want to remove these gobal variables to make it cleaner
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.device object used throughout this script TODO add gpu setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)

    # load the datadump
    tr_data = None
    with open(train_file,'r') as f:
        tr_data = f.readlines()
        for i in range(len(tr_data)):
            tr_data[i] = (tr_data[i][1:len(tr_data[i])-2].split(', '))
            tr_data[i] = [int(j) for j in tr_data[i]]

    # the first list holds vocabulary size
    n_vocab = tr_data[0][0]
    tr_data = tr_data[1:]

    vocab = n_vocab + n_ctx
    
    # set up parameters for the optimizer
    n_train = len(tr_data)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter

    print("n_vocab", n_vocab)
    print("n_ctx", n_ctx)
    print("vocab", vocab)
    print("n_train", n_train, "n_updates_total", n_updates_total)

    # declare the model and lmhead
    model = Model(args, vocab, n_ctx)
    lm_head = LMHead(model, args)

    # declare loss function and the optimizer
    criterion = nn.CrossEntropyLoss(reduce=False) # TODO check loss functions
    model_opt = OpenAIAdam(model.parameters(), lr=lr, schedule=lr_schedule,
                            warmup=lr_warmup, t_total=n_updates_total, b1=b1,
                            b2=b2, e=e, l2=l2, vector_l2=vector_l2,
                            max_grad_norm=max_grad_norm)
    compute_loss_fct = LossCompute(criterion, lm_coef, model_opt)

    # this part will be changed for multigpu support
    model.to(device)
    lm_head.to(device)

    n_updates = 0
    n_epochs = 0

    make_path(os.path.join(save_dir, desc, 'temp.txt'))
    # repeat for n_iter epochs
    while n_epochs < n_iter:
        iters = 0
        # split to train and valid
        _trX, _trV = get_train_valid(tr_data)
        start_ind = 0
        end_ind = start_ind + n_batch_size
        while True:
            cur_batch = _trX[start_ind:end_ind]
            print("epoch ", n_epochs, "iter ", iters)
            trX, trM = transform_code(cur_batch)
            # forward pass and backprop
            run_epoch(trX, trM)
            iters += 1
            start_ind = end_ind
            end_ind = start_ind + n_batch_size
            if end_ind >= len(_trX):
                break
        # validation
        log(_trV)
        n_epochs += 1

        # save the weight pickles
        if submit and n_epochs%3 == 0:
            path = os.path.join(save_dir, desc, 'embed_weights_'+str(n_epochs)+'.pt')
            torch.save(model.embed.weight, path)
    
    if submit:
        path = os.path.join(save_dir, desc, 'embed.pt')
        torch.save(model, path)
