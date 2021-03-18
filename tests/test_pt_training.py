import copy
import unittest

import torch
import torch.nn as nn
from DLBio import pt_training
from DLBio.helpers import check_mkdir
from DLBio.pytorch_helpers import get_device
from DLBio.train_interfaces import Classification
from torch.utils.data import DataLoader, Dataset

import os
import imp

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )

    def forward(self, x):
        return self.f(x)

class TwoDRandomSet(Dataset):
    def __len__(self):
        return 128

    def __getitem__(self, index):
        x = torch.rand(2)
        if x[0] > .5:
            y = 1
        else:
            y = 0
        return x, torch.tensor([y]).long()

class TestRightImport(unittest.TestCase):
    # function name needs to match the patter test_*
    def test_not_imported_from_site_packages(self):
        path = imp.find_module('DLBio')[1]
        tmp = path.split(os.getcwd())[-1]

        self.assertEqual(tmp, '/DLBio')

class TestValidationOnly(unittest.TestCase):

    def test_no_model_change(self):
        model = SimpleModel()
        model_copy = copy.deepcopy(model)

        # setup a training instance to only validate the data
        opt = pt_training.get_optimizer('SGD', model.parameters(), 0.1)
        data_loader = DataLoader(TwoDRandomSet(), 32)

        log_file = 'test_logs/val_only_no_model_change.json'
        check_mkdir(log_file)
        printer = pt_training.get_printer(-1, log_file=log_file)

        train_interface = Classification(
            model, get_device(), printer
        )

        training = pt_training.Training(
            opt, None, train_interface, val_data_loader=data_loader,
            validation_only=True, printer=train_interface.printer
        )

        # train for one epoch
        training(1)

        # the weights of the model must no change when only in validation
        w0_a = model.f[0].weight
        w0_b = model_copy.f[0].weight
        self.assertEqual(torch.abs(w0_a - w0_b).sum(), 0.)

        b0_a = model.f[0].bias
        b0_b = model_copy.f[0].bias
        self.assertEqual(torch.abs(b0_a - b0_b).sum(), 0.)

        w1_a = model.f[2].weight
        w1_b = model_copy.f[2].weight
        self.assertEqual(torch.abs(w1_a - w1_b).sum(), 0.)
        
        b1_a = model.f[2].bias
        b1_b = model_copy.f[2].bias
        self.assertEqual(torch.abs(b1_a - b1_b).sum(), 0.)

if __name__ == '__main__':
    unittest.main()
