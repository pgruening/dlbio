import argparse
import subprocess
import unittest

import torch
import torch.nn as nn
from DLBio import kwargs_translator, pytorch_helpers
from DLBio.helpers import check_mkdir
from DLBio.pt_training import _torch_save_model


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dim', type=int)
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--model_kw', type=str)

    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args)

class SimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim, *, feat_dim):
        super(SimpleModel, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x):
        return self.f(x)

def model_getter(model_type, in_dim, out_dim, device,**kwargs):
    if model_type == 'simple':
        feat_dim = int(kwargs['feat_dim'][-1])
        return SimpleModel(in_dim, out_dim, feat_dim=feat_dim).to(device)
    else:
        raise ValueError

def get_model_fcn(options, device):
    return model_getter(
            'simple', options.in_dim, options.out_dim, device,
            **kwargs_translator.get_kwargs(options.model_kw)
        )

class TestModelLoadingWithOptFile(unittest.TestCase):

    def test_loading(self):
        for save_state_dict in [True, False]:
            for from_json in [True, False]:
                for data_parallel in [True, False]:
                    for strict in [True, False]:
                        kwargs = {
                            'save_state_dict':save_state_dict,
                            'from_json':from_json,
                            'data_parallel':data_parallel,
                            'strict':strict
                        }

                        self._test_correct_loading(**kwargs)
                        self._test_error_opt_does_not_match(**kwargs)
                        self._test_error_no_data_parallel_set(**kwargs)

    def _test_correct_loading(self, *, save_state_dict, from_json, data_parallel, strict):
        original_model, options, device = self._get_model(data_parallel)
        opt_path, model_path = self._save_to_folder(options, original_model, save_state_dict)

        # load the model
        if from_json:
            tmp_opt = opt_path
        else:
            tmp_opt = options

        loaded_model = pytorch_helpers.load_model_with_opt(
            model_path, tmp_opt, get_model_fcn, device,
            strict=strict, from_par_gpu=data_parallel
        )

        if data_parallel:
            original_model = original_model.module
        self._check_same(original_model, loaded_model)

    def _test_error_opt_does_not_match(self, *, save_state_dict, from_json, data_parallel, strict):
        original_model, options, device = self._get_model(data_parallel)

        # options are different from the saved model
        # this must produce an error during loading
        options = get_options([
            '--in_dim', str(1), # should be 2
            '--out_dim',str(3), # should be 2
            '--model_kw', kwargs_translator.to_kwargs_str(
                {'feat_dim': [14]} # should be 32
            )
        ])
        opt_path, model_path = self._save_to_folder(options, original_model, save_state_dict)


        # load the model
        if from_json:
            tmp_opt = opt_path
        else:
            tmp_opt = options

        self.assertRaises(RuntimeError, pytorch_helpers.load_model_with_opt,
            model_path, tmp_opt, get_model_fcn, device,
            strict=strict, from_par_gpu=data_parallel
        )

    def _test_error_no_data_parallel_set(self, *, save_state_dict, from_json, data_parallel, strict):
        if not data_parallel:
            return

        original_model, options, device = self._get_model(data_parallel)
        opt_path, model_path = self._save_to_folder(options, original_model, save_state_dict)

        # load the model
        if from_json:
            tmp_opt = opt_path
        else:
            tmp_opt = options
        
        # if the model is save in parallel mode, but the from_par_gpu flag is not set,
        # an error needs to be raised
        self.assertRaises(RuntimeError, pytorch_helpers.load_model_with_opt,
            model_path, tmp_opt, get_model_fcn, device,
            strict=strict, from_par_gpu=False
        )


    def _get_model(self, data_parallel):
        options = get_options([
            '--in_dim', str(2),
            '--out_dim',str(2),
            '--model_kw', kwargs_translator.to_kwargs_str(
                {'feat_dim': [32]}
            )
        ])

        device = pytorch_helpers.get_device()
        original_model = get_model_fcn(options, device)

        if data_parallel:
            original_model = nn.DataParallel(original_model)

        return original_model, options, device

    def _save_to_folder(self, options, original_model, save_state_dict):
        # save options
        opt_path = 'test_logs/model_saving/opt.json'
        check_mkdir(opt_path)
        pytorch_helpers.save_options(opt_path, options)

        # save the model
        model_path = 'test_logs/model_saving/model.pt'
        _torch_save_model(original_model, model_path, save_state_dict=save_state_dict)
        
        return opt_path, model_path


    def _check_same(self, original_model, new_model):
        w0_a = original_model.f[0].weight
        w0_b = new_model.f[0].weight
        self.assertEqual(torch.abs(w0_a - w0_b).sum(), 0.)

        b0_a = original_model.f[0].bias
        b0_b = new_model.f[0].bias
        self.assertEqual(torch.abs(b0_a - b0_b).sum(), 0.)

        w1_a = original_model.f[2].weight
        w1_b = new_model.f[2].weight
        self.assertEqual(torch.abs(w1_a - w1_b).sum(), 0.)
        
        b1_a = original_model.f[2].bias
        b1_b = new_model.f[2].bias
        self.assertEqual(torch.abs(b1_a - b1_b).sum(), 0.)

if __name__ == '__main__':
    unittest.main()
