import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from DLBio.helpers import check_mkdir, load_json
from DLBio import pt_training
from DLBio.train_interfaces import Classification
from DLBio.pytorch_helpers import get_device
NUM_CLASSES = 10


class FakeModelAlwaysRight(nn.Module):
    def __init__(self, device):
        super(FakeModelAlwaysRight, self).__init__()
        self.unused_conv = nn.Conv2d(1, 1, 1)
        self.d = device

    def forward(self, x):
        out = torch.zeros(x.shape[0], NUM_CLASSES).to(self.d)
        for i in range(x.shape[0]):
            out[i, int(x[i])] = 10.

        return out


class DatasetXisY(Dataset):
    def __init__(self, ds_len):
        self.len = ds_len
        self.data = torch.round((NUM_CLASSES-1) * torch.rand(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.data[index]
        return x, x.long()


class TestClassificationMetrics(unittest.TestCase):

    def test_metrics(self):
        num_val_data = 71
        print(f'Number of samples used: {num_val_data}')

        model = FakeModelAlwaysRight(get_device()).to(get_device())
        val_data_loader = DataLoader(DatasetXisY(num_val_data), 32)

        opt = pt_training.get_optimizer('SGD', model.parameters(), 0.1)

        log_file = 'test_logs/train_interface_model_alway_right.json'
        check_mkdir(log_file)
        printer = pt_training.get_printer(-1, log_file=log_file)

        train_interface = Classification(
            model, get_device(), printer
        )
        train_interface.functions['acc'].print_len = True

        training = pt_training.Training(
            opt, None, train_interface, val_data_loader=val_data_loader,
            validation_only=True, printer=train_interface.printer
        )

        training(3)

        log = load_json(log_file)

        self.assertEqual(log['val_acc'][-1], 1.)
        self.assertEqual(log['val_er'][-1], 0.)
        self.assertEqual(log['val_num_samples'][-1], num_val_data)


if __name__ == '__main__':
    unittest.main()
