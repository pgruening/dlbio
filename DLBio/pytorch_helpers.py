import json
from collections import OrderedDict

import numpy as np
import torch
from recordtype import recordtype

from .helpers import dict_to_options, load_json


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
        raise ValueError('Unknown dimensionality: {}'.format(x.dim()))


def image_batch_to_tensor(x):
    assert x.dtype == 'uint8'
    assert x.ndim == 4
    x = torch.Tensor(x.astype('float32') / 255.)
    x = x.permute([0, 3, 1, 2])
    return x


def check_norm(model):
    P = ctr_ = 0.0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        P += p.grad.data.norm(2).item()
        ctr_ += 1.0

    print('Mean gradient: {}'.format(P / ctr_))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_model_with_opt(model_path, options, get_model_fcn, device, strict=False, map_location=None, from_par_gpu=False):
    """Load a model with pt-file located at model_path and and options file
    located at options_path, using get_model_fcn.

    In your specific repo, you should create a 'load_model' function, that
    implements the specific get_model_fcn and uses this function to handle
    the pytorch part. 

    Parameters
    ----------
    model_path : str
        path to a pytorch model-file: [name].pt, if is None, the model
        is only loaded from scratch
    options : str / or object
        path to a json file that contains an options dictionary, or the options
        object itself
    get_model_fcn : function(NamedTuple, str)
        functions that, given an options object and a device loads a model
        (from scratch)
    device : str
        Defines whether a cpu ('cpu') or gpu ('cuda:0') is used.

    Returns
    -------
    nn.Module
        a pytorch model with the weights taken from model_path
    """

    if isinstance(options, str):
        options = load_json(options)
        options = dict_to_options(options)

    model = get_model_fcn(options, device)

    if model_path is None:
        return model

    if map_location is not None:
        model_sd = torch.load(model_path, map_location=map_location)
    else:
        model_sd = torch.load(model_path)

    if not isinstance(model_sd, OrderedDict):
        model_sd = model_sd.state_dict()
        if from_par_gpu:
            raise NotImplementedError('TODO...')
    else:
        if from_par_gpu:
            new_dict = OrderedDict()
            for key, value in model_sd.items():
                new_key = key.replace('module.', '')
                new_dict[new_key] = value
            model_sd = new_dict

    model.load_state_dict(model_sd, strict=strict)

    return model
