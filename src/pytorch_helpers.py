import json

import numpy as np
import torch
from recordtype import recordtype


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def get_num_params(model, count_only_trainable=True):
    def select(p):
        return p.requires_grad or not count_only_trainable

    model_p = [p for p in model.parameters() if select(p)]
    #model_p = list(model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_p])
    return num_params


def get_options_object_from_file(opt_file):
    with open(opt_file, 'r') as opt_file:
        options = json.load(opt_file)

    OptionObject = recordtype('OptionObject', options.keys())
    options = OptionObject(**options)

    return options


def cuda_to_numpy(x):
    x = x.cpu().detach()
    if x.dim() == 4:
        return x.permute([0, 2, 3, 1]).numpy()
    elif x.dim() == 3:
        return x.permute([1, 2, 0]).numpy()
    else:
        raise ValueError('Unkwnown dimensionality: {}'.format(x.dim()))


def check_norm(model):
    P = ctr_ = 0.0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        P += p.grad.data.norm(2).item()
        ctr_ += 1.0

    print('Mean gradient: {}'.format(P / ctr_))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
