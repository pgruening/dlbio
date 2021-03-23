import json
import subprocess
from collections import OrderedDict
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from recordtype import recordtype

from .helpers import dict_to_options, load_json


def get_free_gpus(thres=1024):
    """Returns a list of gpu-indices, where the memory allocation does not 
    exceed thres.

    Parameters
    ----------
    thres : int, optional
        number of MiB. If this number is exceeded the gpu is considere 
        'not-free' , by default 1024

    Returns
    -------
    list of int
        indices of GPU that are free to use
    """
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.used'] = gpu_df['memory.used'].map(
        lambda x: int(x.rstrip(' [MiB]')))

    free_gpu_indices = [
        i for i, x in enumerate(list(gpu_df['memory.used'])) if x < thres
    ]
    return free_gpu_indices


class GetBlocks():
    """
    use like this:
    get_blocks = GetBlocks()
    found_blocks = get_blocks(model, (module1, module2, module3))

    Don't forget to reset the counter if you want to reuse the function
    """

    def __init__(self):
        self.ctr = 0

    def __call__(self, module, used_blocks):
        """Returns all object that are of a type given in used_blocks and
        contained in module (e.g. a CNN model).

        Parameters
        ----------
        module : nn.Module
            a module containing other modules
        used_blocks : tuple of classes (typically nn.Module)
            what classes to look for

        Returns
        -------
        list of tuples: (int, object)
            int = depth (number of convolution before the module)
            object = the actual module with type in used_blocks
        """
        assert isinstance(used_blocks, tuple)
        out = []
        if isinstance(module, used_blocks):
            self.ctr += 1
            out.append([self.ctr, module])
        else:
            if isinstance(module, nn.Conv2d):
                self.ctr += 1

            for child in module.children():
                out += self(child, used_blocks)

        return out


class ActivationGetter():
    """use like this
    get_conv_activation = ActivationGetter(model.conv1)
    y = model(x)
    conv_activation = get_conv_activation.out
    """

    def __init__(self, module):
        """A way to save the outputs of specific modules in a bigger model,
        e.g., if you want to check outputs of specific layers.

        Parameters
        ----------
        module : nn.Module
            the forward output of the module is written in the variable "out"
            after using the forward pass.
        """
        self.hook = module.register_forward_hook(self._hook_fcn)
        self.out = None

    def _hook_fcn(self, module, input, output):
        self.out = output


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def save_options(file_path, options):
    if not hasattr(options, "__dict__"):
        out_dict = dict(options._asdict())
    else:
        out_dict = options.__dict__

    # add the current time to the output
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

    out_dict.update({
        'start_time': dt_string
    })

    # add the used pytorch version, with it, it is easier to identity
    # problems when loading a model
    out_dict['torch_version'] = torch.__version__

    with open(file_path, 'w') as file:
        json.dump(out_dict, file)


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


def load_model_with_opt(model_path, options, get_model_fcn, device, strict, map_location=None, from_par_gpu=False):
    """Load a model with pt-file located at model_path and and options file
    located at options_path, using get_model_fcn.

    In your specific repo, you should create a 'load_model' function, that
    implements the specific get_model_fcn and uses this function to handle
    the pytorch part.

    Note that only the state dict is loaded here. Even if the model as saved
    as an entire model (torch.save(model, path))!

    This function is only meant for strictly loading the same architecture.
    If you want to somehow only want to load a subset or weights from a
    different model, write a costum function.

    Parameters
    ----------
    model_path : str
        path to a pytorch model-file: [name].pt, if is None, the model
        is only loaded from scratch
    options : str / or object
        path to a json file that contains an options dictionary, or the options
        object itself
    get_model_fcn : function(NamedTuple, str)
        functions that, given an options object and a device, loads a model
        (from scratch)
    device : str
        Defines whether a cpu ('cpu') or gpu ('cuda:0') is used.
    strict: boolean
        Whether you are loading from a partial state_dict, 
        which is missing some keys, or loading a state_dict with more keys
        than the model that you are loading into, you can set the strict 
        argument to False.

    Returns
    -------
    nn.Module
        a pytorch model with the weights taken from model_path
    """

    def _change_key_name_from_data_parallel(model_sd):
        new_dict = OrderedDict()
        for key, value in model_sd.items():
            new_key = key.replace('module.', '')
            new_dict[new_key] = value
        return new_dict

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
            model_sd = _change_key_name_from_data_parallel(model_sd)
    else:
        if from_par_gpu:
            model_sd = _change_key_name_from_data_parallel(model_sd)

    x = model.load_state_dict(model_sd, strict=strict)
    if x.missing_keys:
        raise RuntimeError('Missing keys detected')

    return model
