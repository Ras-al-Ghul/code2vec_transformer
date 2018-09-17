import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import math, sys, os

from tensorflow.python.framework import function
from termcolor import colored
from functools import partial

tf.logging.set_verbosity(tf.logging.INFO)
# prints entire np array
np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='code')
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--save_dir', type=str, default='save/')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--submission_dir', type=str, default='submission/')
parser.add_argument('--submit', action='store_true')
parser.add_argument('--analysis', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_iter', type=int, default=3)
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--n_batch', type=int, default=8)
parser.add_argument('--n_batch_size', type=int, default=2000000)    # num of lines of code in a batch
parser.add_argument('--max_grad_norm', type=int, default=1)
parser.add_argument('--lr', type=float, default=6.25e-5)
parser.add_argument('--lr_warmup', type=float, default=0.002)
parser.add_argument('--n_ctx', type=int, default=22)
parser.add_argument('--n_embd', type=int, default=36)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--vector_l2', action='store_true')
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--afn', type=str, default='gelu')
parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
parser.add_argument('--src_folder', type=str, default='./applet')
parser.add_argument('--train_file', type=str, default='../Transformer_Code_Attention/lex_dumps_data.txt')
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument('--b1', type=float, default=0.9)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--e', type=float, default=1e-8)

args = parser.parse_args()
print(args)
globals().update(args.__dict__)

# collect the data from the dump
tr_data = None
with open(train_file,'r') as f:
    tr_data = f.readlines()
    for i in range(len(tr_data)):
        tr_data[i] = (tr_data[i][1:len(tr_data[i])-2].split(', '))
        tr_data[i] = [int(j) for j in tr_data[i]]

n_vocab = tr_data[0][0]
tr_data = tr_data[1:]
temp_dict = {'n_train':len(tr_data)}
globals().update(temp_dict)
n_updates_total = (n_train//n_batch)*n_epoch
temp_dict = {'n_updates_total':n_updates_total}
globals().update(temp_dict)
print(colored("total number of lines of code "+str(n_train), 'green'))

# print utility
# needed as we need to insert a node in the tf graph for printing
# refer https://towardsdatascience.com/using-tf-print-in-tensorflow-aa26e1cff11e
def my_func(x):
    # import boto3

    # # Method 1: Object.put()
    # s3 = boto3.resource('s3')
    # global ct
    # if ct%10 == 0:
    #     object = s3.Object('sagemaker-us-east-1-082830052325', 'temp/filename-'+str(ct)+'.txt')
    #     object.put(Body=str(x))
    #     print(ct)
    # ct += 1
    print(colored(str(x),'yellow'))
    return x

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

# collects all the trainable variables for backprop
def find_trainable_variables(key):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))

def get_ema_vars(*vs):
    if tf.get_variable_scope().reuse:
        gvs = tf.global_variables()
        vs = [get_ema_if_exists(v, gvs) for v in vs]
    if len(vs) == 1:
        return vs[0]
    else:
       return vs

# gelu activation
def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

# swish activation
def swish(x):
    return x*tf.nn.sigmoid(x)

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

# standard warmup functions for the optimizer
def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))

def warmup_constant(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1

def warmup_linear(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)

schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

# declaring the Adam Optimizer with weight decay
class AdamOp():
    def __init__(self):
        pass

    def minimize(self, params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False, max_grad_norm=-1, **kwargs):
        """
        adam with weight decay fix
        """
        t = tf.Variable(0, dtype=tf.float32, trainable=False)
        tt = t+1
        updates = [t.assign(tt)]
        if max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        for p, g in zip(params, grads):
            if p is None or g is None:
                print("can't train", p.name, g)
            else:
                if isinstance(g, tf.IndexedSlices):
                    g = tf.convert_to_tensor(g)
                m = tf.Variable(p*0, dtype=tf.float32, trainable=False)
                v = tf.Variable(p*0, dtype=tf.float32, trainable=False)
                lrt = lr*tf.sqrt(1-b2**tt)/(1-b1**tt)
                lrt *= schedule(t/t_total)
                mt = b1*m + (1-b1)*g
                vt = b2*v + (1-b2)*g*g
                if (len(p.get_shape()) > 1 or vector_l2) and l2 > 0:
                    pt = p - lrt * (mt / (tf.sqrt(vt) + e) + l2*p)
                else:
                    pt = p - lrt * (mt / (tf.sqrt(vt) + e))
                updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
        return tf.group(*updates)
    

# opt_fns = {
#     'adam':adam,
# }

# the layer norms - every layer has a add & norm
def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

# randomly not using some neurons
def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

# sub method for split_heads
def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

# sub method for merge_heads
def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

# split heads for multiheadattention
def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

# merge heads after multiheadattention
def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

# convolution for the position wide feed forward network
def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

# the attention sublayer
def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

# mlp - feed forward layer
def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

# a block of the transformer model
def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

# the embedding layer
def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

# For Sagemaker
# declare the model and the loss function
# also use the optimizer to backprop the loss
def model_fn(features, labels, mode, params):
    with tf.variable_scope('model'):
        we = tf.get_variable("we", [n_vocab+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, mode == tf.estimator.ModeKeys.TRAIN)

        features = tf.reshape(features, [-1, n_ctx, 2])
        labels = tf.reshape(labels, [-1, n_ctx])

        h = embed(features, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=(mode == tf.estimator.ModeKeys.TRAIN), scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(features[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(features)[0], shape_list(features)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*labels[:, 1:], 1)/tf.reduce_sum(labels[:, 1:], 1)
        lm_losses = tf.reduce_mean(lm_losses)

        lm_losses = tf.Print(lm_losses, [lm_losses, we.shape])
        print(we.shape, "we")

    # calling custom Adam optimizer instead of tf.train.AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    # optimizer = AdamOp()
    train_op = optimizer.minimize(loss=lm_losses, global_step=tf.train.get_global_step())
    # params = find_trainable_variables("model")
    # grads = tf.gradients(lm_losses, params)
    # train_op = optimizer.minimize(params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {}
    
    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=lm_losses,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops, predictions={})

# For Sagemaker
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(save_summary_steps=1)

# For Sagemaker
language_model = tf.estimator.Estimator(model_fn=model_fn, params={"learning_rate": 0.00001,
                                                                },
                                                                config=run_config)
# set up the number of training steps
n_steps = int(len(tr_data)*n_epoch/n_batch)
n_steps = 100
print(colored("Running for "+str(n_steps)+" steps", 'red'))

# transform code with positional embeddings and set bitmasks
def transform_code(X):
    n_size = len(X)
    xmb = np.zeros((n_size, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_size, n_ctx), dtype=np.float32)
    for i, x, in enumerate(X):
        l = len(x)
        xmb[i, :l, 0] = x
        mmb[i, :l] = 1
    xmb[:, :, 1] = np.arange(n_vocab, n_vocab+n_ctx)
    return xmb, mmb

# trX is actual data, trM is bitmask
trX, trM = transform_code(tr_data)

# For Sagemaker, cannot be renamed
def train_input_fn(training_dir, params):
    return tf.estimator.inputs.numpy_input_fn(x={'inputs': trX},
                                              y=trM,
                                              num_epochs=None,
                                              shuffle=False)()

# for testing the estimator locally
def train_input_fn_local(features, labels, local_batch_size):
    """An input function for training, to be used by the estimator on a local system (This is not the sagemaker input function)"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(local_batch_size).repeat()
    return dataset

# For Sagemaker
train_result = language_model.train(input_fn=lambda: train_input_fn_local(trX, trM, n_batch), steps=n_steps)
print("train complete")