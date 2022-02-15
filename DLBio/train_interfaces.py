
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from .pt_train_printer import IPrinterFcn
from .pt_training import ITrainInterface

# ----------------------------------------------------------------------------
# -------------------------------CLASSIFICATION.------------------------------
# ----------------------------------------------------------------------------


class Classification(ITrainInterface):
    name = 'classification'

    def __init__(self, model, device, printer):
        self.printer = printer
        self.model = model
        self.xent_loss = nn.CrossEntropyLoss()
        self.functions = {
            'acc': Accuracy(),
            'er': ErrorRate()
        }
        self.counters = {
            'num_samples': image_counter
        }
        self.d = device

    def train_step(self, sample):
        images, targets = sample[0].to(self.d), sample[1].to(self.d)
        pred = self.model(images)

        if targets.ndim == 2:
            assert targets.shape[-1] == 1, f'Unknown target shape {targets.shape}'
            targets = targets[:, 0]

        loss = self.xent_loss(pred, targets)
        assert not bool(torch.isnan(loss))

        metrics = None
        counters = {}
        counters.update({k: v(pred, targets)
                         for k, v in self.counters.items()})
        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return loss, metrics, counters, functions


class Accuracy(IPrinterFcn):
    name = 'acc'

    def __init__(self, debug_print_len=False):
        self.print_len = debug_print_len
        self.restart()

    def update(self, y_pred, y_gt):
        assert y_pred.ndim == 2

        assert y_gt.ndim <= 2
        if y_gt.ndim == 2:
            assert y_gt.shape[-1] == 1

        # get class with highest value
        y_pred = np.array(y_pred.detach().cpu())
        y_pred = np.argmax(y_pred, 1)
        self.x.append(y_pred)
        self.y.append(np.array(y_gt.detach().cpu()).flatten())
        return self

    def restart(self):
        self.name = type(self).name
        self.x = []
        self.y = []

    def __call__(self):
        x = np.concatenate(self.x)
        y = np.concatenate(self.y)

        if self.print_len:
            self._print_len(x, y)

        return (x == y).mean()

    def _print_len(self, x, y):
        print(f'Length self.x: {len(self.x)}')
        print(f'Length self.y: {len(self.y)}')

        print(f'shape x: {x.shape}')
        print(f'shape y: {y.shape}')


class ErrorRate(Accuracy):
    name = 'e_rate'

    def __init__(self, debug_print_len=False):
        self.print_len = debug_print_len
        self.restart()

    def __call__(self):
        acc = super(ErrorRate, self).__call__()
        return (1. - acc) * 100.


def image_counter(y_pred, y_true):
    return float(y_true.shape[0])


# ----------------------------------------------------------------------------
# -------------------------------REGRESSION-----------------------------------
# ----------------------------------------------------------------------------
class Regression(ITrainInterface):
    name = 'regression'

    def __init__(self, model, device, printer, use_sigmoid=False, loss_type='l1'):
        self.printer = printer
        self.model = model
        if loss_type == 'l1':
            self.loss_fcn = nn.L1Loss()
        elif loss_type == 'mse':
            self.loss_fcn = nn.MSELoss()

        self.functions = {
            'pearson': PearsonCorrelation(),
            'spearman': SpearmanCorrelation(),
        }
        self.counters = {
            'num_samples': image_counter
        }
        self.d = device
        self.use_sigmoid = use_sigmoid

    def train_step(self, sample):
        images, targets = all_to(sample[0], self.d), sample[1].to(self.d)
        pred = self.model(images)
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)

        if targets.ndim == 2:
            assert targets.shape[-1] == 1, (
                f'Unknown target shape {targets.shape}')
            targets = targets[:, 0]

        if pred.ndim == 2:
            assert pred.shape[-1] == 1, (
                f'Unknown prediction shape {pred.shape}')
            pred = pred[:, 0]

        loss = self.loss_fcn(pred, targets)
        assert not bool(torch.isnan(loss))

        metrics = None
        counters = {}
        counters.update({k: v(pred, targets)
                         for k, v in self.counters.items()})
        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return loss, metrics, counters, functions


class ICorrelationFunction(IPrinterFcn):
    name = None

    def __init__(self):
        self.restart()

    def update(self, y_pred, y_gt):
        self.x.append(np.array(y_pred.detach().cpu()).flatten())
        self.y.append(np.array(y_gt.detach().cpu()).flatten())
        return self

    def restart(self):
        self.name = type(self).name
        self.x = []
        self.y = []

    def _corr_fcn(self, x, y):
        raise NotImplementedError()

    def __call__(self):
        x = np.concatenate(self.x, -1)
        y = np.concatenate(self.y, -1)
        return self._corr_fcn(x, y)


class PearsonCorrelation(ICorrelationFunction):
    name = 'pearson'

    def _corr_fcn(self, x, y):
        return pearsonr(x, y)[0]


class SpearmanCorrelation(ICorrelationFunction):
    name = 'spearman'

    def _corr_fcn(self, x, y):
        return spearmanr(x, y)[0]


def all_to(X, device):
    if isinstance(X, (list, tuple)):
        out = []
        for x in X:
            if isinstance(x, torch.Tensor):
                out.append(x.to(device))
            else:
                out.append(x)

        return out
    else:
        return X.to(device)
