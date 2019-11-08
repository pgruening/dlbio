import os
import random

import cv2
import numpy as np

import torch
import torch.optim as optim
from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_lr
from pytorch_lamb import Lamb


class ITrainInterface():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Needs model and loss fcn and metrics')

    def train_step(self, *args, **kwargs):
        raise NotImplementedError('Implement to run training')


class Training():
    def __init__(
            self, optimizer, data_loader, train_interface,
            save_steps=-1, output_path=None,
            metric_fcns_=[], printer=None, scheduler=None, clip=None
    ):
        self.optimizer = optimizer
        self.data_loader = data_loader

        assert issubclass(train_interface.__class__, ITrainInterface)
        self.train_interface = train_interface

        self.metric_fcns_ = metric_fcns_
        self.scheduler = scheduler

        if printer is None:
            self.printer = Printer(100, None)
        else:
            self.printer = printer

        if save_steps > 0:
            assert output_path is not None

        self.do_save = save_steps > 0 and output_path is not None
        self.save_steps = save_steps
        self.save_path = output_path

        self.clip = clip

    def __call__(self, epochs_):
        self.printer.restart()

        for epoch in range(epochs_):
            self.printer.learning_rate = get_lr(self.optimizer)

            for sample in self.data_loader:
                loss, metrics = self.train_interface.train_step(sample)
                self._update_weights(epoch, loss, metrics)

            self.printer.on_epoch_end()

            self._schedule()
            self._save(epoch, epochs_)

    def _update_weights(self, epoch, loss, metrics):
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.train_interface.model.parameters(), self.clip
            )

        self.optimizer.step()

        self.printer.update(loss, epoch, metrics)
        self.printer.print_conditional()

    def _schedule(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def _save(self, epoch, epochs_):
        if self.do_save:
            if epoch == epochs_ - 1 or epoch % self.save_steps == 1:
                torch.save(self.train_interface.model, self.save_path)


def get_optimizer(opt_id, parameters, learning_rate, **kwargs):
    if opt_id == 'SGD':
        optimizer = optim.SGD(parameters,
                              lr=learning_rate,
                              momentum=kwargs.get('momentum', .9))
    elif opt_id == 'Adam':
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate
        )
    elif opt_id == 'lamb':
        optimizer = Lamb(
            parameters,
            lr=learning_rate, weight_decay=kwargs.get('weight_decay', 0.001),
            betas=(kwargs.get('beta0', .9), kwargs.get('beta1', .999))
        )
    else:
        raise ValueError(f'Unknown opt value: {opt_id}')

    return optimizer


def get_scheduler(lr_steps, epochs, optimizer, gamma=.1):
    step_size = epochs // lr_steps
    print(f'Sched step size: {step_size}')

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size,
        gamma=gamma, last_epoch=-1
    )

    return scheduler


def set_device(device=None):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        print(f'using device {device}')


def set_random_seed(seed):
    print(f'Setting seed: {seed}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    # for debugging purposes some random numbers are generated
    output = {
        'seed': seed,
        'torch': torch.randn(1).item(),
        'cuda': torch.cuda.FloatTensor(1).normal_().item(),
        'numpy': float(np.random.randn(1)),
        'python': random.randint(0, 5000)
    }
    # with open(os.path.join(options.folder_name, 'rand_num_test.json'), 'w') as file:
    #    json.dump(output, file)

    return _init_fn
