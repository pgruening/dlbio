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
        # NOTE: needs a model a loss function and metrics
        raise NotImplementedError('Needs model and loss fcn and metrics')

    def train_step(self, *args, **kwargs):
        # NOTE use like this:
        # loss, metrics = self.train_interface.train_step(sample)
        raise NotImplementedError('Implement to run training')


class Training():
    def __init__(
            self, optimizer, data_loader, train_interface,
            save_steps=-1, output_path=None,
            metric_fcns_=[], printer=None, scheduler=None, clip=None,
            retain_graph=False, val_data_loader=None, early_stopping=None
    ):
        self.optimizer = optimizer
        self.data_loader = data_loader

        assert issubclass(train_interface.__class__, ITrainInterface)
        self.train_interface = train_interface

        self.metric_fcns_ = metric_fcns_
        self.scheduler = scheduler
        self.early_stopping = early_stopping

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
        self.retain_graph = retain_graph

        self.phases = ['train']
        if val_data_loader is not None:
            self.phases.append('validation')

        self.data_loaders_ = {'train': data_loader,
                              'validation': val_data_loader}

    def __call__(self, epochs_):
        self.printer.restart()

        do_stop = False

        for epoch in range(epochs_):
            self.printer.learning_rate = get_lr(self.optimizer)

            for current_phase in self.phases:
                if current_phase == 'train':
                    self.train_interface.model.train()
                else:
                    self.train_interface.model.eval()

                for sample in self.data_loaders_[current_phase]:

                    loss, metrics = self._train_step(sample, current_phase)
                    self._update_printer(epoch, loss, metrics, current_phase)

                    if current_phase == 'train':
                        self._update_weights(loss)

                if self.early_stopping is not None and current_phase == 'validation':
                    do_stop = self.early_stopping(
                        self.printer.get_metrics(),
                        self.train_interface.model,
                        self.save_path
                    )
                self.printer.on_epoch_end()

                self._schedule(current_phase)
                self._save(epoch, epochs_)

            if do_stop:
                return

    def _train_step(self, sample, current_phase):
        if current_phase == 'validation':
            with torch.no_grad():
                loss, metrics = self.train_interface.train_step(sample)
        else:
            loss, metrics = self.train_interface.train_step(sample)
        return loss, metrics

    def _update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=self.retain_graph)

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.train_interface.model.parameters(), self.clip
            )

        self.optimizer.step()

    def _update_printer(self, epoch, loss, metrics, current_phase):
        if current_phase == 'train':
            self.printer.update(loss, epoch, metrics)
        else:
            if metrics is not None:
                metrics = {'val_' + k: v for (k, v) in metrics.items()}
            self.printer.update(loss, epoch, metrics, loss_key='val_loss')

        self.printer.print_conditional()

    def _schedule(self, current_phase):
        if self.scheduler is not None:
            if current_phase == 'train':
                self.scheduler.step()

    def _save(self, epoch, epochs_):
        if self.do_save:
            if epoch == epochs_ - 1 or epoch % self.save_steps == 1:
                print(f'Saving {self.save_path}')
                torch.save(self.train_interface.model, self.save_path)


def get_optimizer(opt_id, parameters, learning_rate, **kwargs):
    if opt_id == 'SGD':
        optimizer = optim.SGD(parameters,
                              lr=learning_rate,
                              momentum=kwargs.get('momentum', .9),
                              weight_decay=kwargs.get('weight_decay', 0.)
                              )
    elif opt_id == 'Adam':
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0.)
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
    assert lr_steps < epochs, f'Epochs must be greater than lr_steps but e:{epochs} < l:{lr_steps}'
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


def loss_verification(train_interface, data_loader, printer):
    # verify loss
    print('Running loss verification')
    with torch.no_grad():
        mean_im = 0.0
        std_im = 0.0
        ctr = 0.0
        for sample in data_loader:
            mean_im += sample['x'].mean()
            std_im += sample['x'].std()
            ctr += 1.0
            loss, metrics = train_interface.train_step(sample)
            printer.update(loss, -1, metrics)

        printer.print()
        print(f'mean: {mean_im/ctr:.3f} std: {std_im/ctr:.3f}')


class EarlyStopping():
    def __init__(self, metric_key, get_max=True, epoch_thres=np.inf):
        self.key = metric_key
        self.get_max = get_max

        self.no_update_counter = 0.
        self.thres = epoch_thres

        if get_max:
            self.current_val = -np.inf
        else:
            self.current_val = +np.inf

    def __call__(self, metrics, model, output_path):
        value = metrics[self.key]

        self.no_update_counter += 1
        if self.get_max:
            if value > self.current_val:
                self._update(value, model, output_path)
        else:
            if value < self.current_val:
                self._update(value, model, output_path)

        if self.no_update_counter > self.thres:
            return True
        else:
            return False

    def _update(self, value, model, output_path):
        self.no_update_counter = 0
        self.current_val = value
        torch.save(model, output_path)
