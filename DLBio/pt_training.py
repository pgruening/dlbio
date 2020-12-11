import argparse
import os
import random
import time
import warnings
from math import cos, pi

import cv2
import numpy as np
import torch
import torch.optim as optim
from pytorch_lamb import Lamb

from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_lr


def get_train_arg_parser(config):
    """Typical argument parser to train a neural network

    Parameters
    ----------
    config : module or object
        default values for your project

    Returns
    -------
    argument parser
        use like this:
        import config_module
        ...
        ...
        def get_options():
            parser = get_train_argparser(config_module)
            parser.add_argument(...)
            ...

            return parser.parse_args()

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--wd', type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--mom', type=float, default=config.MOMENTUM)
    parser.add_argument('--opt', type=str, default=config.OPTIMIZER)

    parser.add_argument('--bs', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr_steps', type=int, default=config.LR_STEPS)
    parser.add_argument('--nw', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--sv_int', type=int, default=config.SAVE_INTERVALL)
    parser.add_argument('--model_type', type=str, default=config.MODEL_TYPE)

    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--device', type=int, default=config.DEVICE)

    parser.add_argument('--folder', type=str, default=config.DEF_FOLDER)
    parser.add_argument('--model_name', type=str, default=config.MODEL_NAME)

    parser.add_argument('--in_dim', type=int, default=config.INPUT_DIM)

    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--es_metric', type=str, default=config.ES_METRIC)

    parser.add_argument('--num_classes', type=int, default=config.NUM_CLASSES)

    # may be unnecessary for your project
    parser.add_argument('--ds_len', type=int, default=config.DATASET_LENGTH)
    parser.add_argument('--crop_size', type=int, default=config.CROP_SIZE)

    return parser


class ITrainInterface():
    """
    Train Interface handle the prediction of the network, the loss 
    computation and the computation of additional training metrics.
    These steps can quickly change depending on the dataset, the model
    architecture, the task and so on. Therefore, it is reasonable to
    create separate modules that are passed to the Training class.

    You need to implement the constructor and the train_step method,
    if the computations in the validation step differ from the train_step
    you need to overwrite val_step

    """

    def __init__(self, *args, **kwargs):
        """Constructor. Usually you need to provide and process:
        - a model
        - a device
        - implement a loss function
        - implement additional metrics
        """
        raise NotImplementedError('Needs model and loss fcn and metrics')

    def train_step(self, *args, **kwargs):
        """
        In the Training class, this functions is called for each drawn batch
        like this:

        loss, metrics = self.train_interface.train_step(sample)
        (for more information see '_train_step' method)

        Accordingly, you should compute the loss based on the prediction of
        your model and other metrics.

        The loss is used to update the weights of the model

        """
        raise NotImplementedError('Implement to run training')

    def val_step(self, *args, **kwargs):
        """
        By default, the same code as in train_step is excecuted.
        """
        # usually exactly the same as the train step
        return self.train_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        """
        By default, the same code as in val_step is excecuted.
        """
        # usually exactly the same as the train step
        return self.val_step(*args, **kwargs)


class Training():
    """Class that contains all necessary ingredients to train a pytorch
    model. To start training simple call the instantiated object with the
    desired number of epoch

    e.g.:
    training = Training(...)
    training(100) # train for 100 epochs


    """

    def __init__(
            self, optimizer, data_loader, train_interface,
            save_steps=-1, save_path=None,
            printer=None, scheduler=None, clip=None,
            retain_graph=False, val_data_loader=None, early_stopping=None,
            validation_only=False, save_state_dict=False,
            test_data_loader=None, batch_scheduler=None, start_epoch=0,
            time_log_printer=None
    ):
        """Constructor

        Parameters
        ----------
        optimizer : pytorch optimizer
            controls the weight updates, see get_optimizer for more information
        data_loader : pytorch dataloader
            when iterated over in a for loop, data are returned in batches.
            Note that the for loop is executed as
            'for sample in data_loader:'
            You need to specify what sample actually is in the training-
            interface.
        train_interface : ITrainInterface
            computes the loss of a batch, see method _train_step
        save_steps : int, optional
            every 'save_steps' the model is save to 'save_path'. If 0, the
            model is only saved on the end of the training. By default -1,
            which means the model is not saved at all (if early_stopping is 
            None).
        save_path : str, optional
            where to save the model, by default None. Needs to be specified if
            save_steps != 1
        printer : Printer (pt_train_printer), optional
            Prints current training values to terminal and possibly a
            log.json file. By default None, nothing is printed or logged.
        scheduler : pytorch scheduler, optional
            updates the learning rate according to some schedule. By default 
            None, no scheduling is used.
        clip : float, optional
            gradient clipping, by default None, no gradient clipping
        retain_graph : bool, optional
            needed for special backpropagation function, see pytorch 
            documentation for more information. By default False.
        val_data_loader : pytorch data_loader, optional
            can be used to validation the network performance. These data are
            not used for training (but maybe early stopping). The model is in
            eval-mode, when those data are applied.
            By default None, no validation is done.
        early_stopping : EarlyStopping object, optional
            save the model based on a specified metric, each time the best 
            value of this metric is reached. By default None, no early stopping
        validation_only: bool 
            when called, only the validation steps are computed
        save_state_dict: save the model's state dict instead of the model
        batch_scheduler: BatchScheduler object
            for scheduling algorithms that adjust the learning
            rate within an epoch, instead each epoch's end.
        start_epoch: int 
            set to a value other than 0 if training is resumed
        time_log_printer: Printer (pt_train_printer)
            if not none, several the time needed for different training steps
            is logged and written by this logger

        Returns
        -------
        Training object
        """
        self.optimizer = optimizer
        self.data_loader = data_loader

        assert issubclass(train_interface.__class__, ITrainInterface)
        self.train_interface = train_interface

        self.scheduler = scheduler
        self.batch_scheduler = batch_scheduler
        self.early_stopping = early_stopping

        if printer is None:
            self.printer = Printer(100, None)
        else:
            self.printer = printer

        self.time_log_printer = time_log_printer
        self.time_logger = TimeLogger(is_active=(time_log_printer is not None))

        assert isinstance(save_steps, int)
        if save_steps > 0:
            assert save_path is not None

        self.do_save = save_steps > 0 and save_path is not None
        self.save_steps = save_steps
        self.save_path = save_path
        self.save_state_dict = save_state_dict
        print(self.save_state_dict)

        self.clip = clip
        self.retain_graph = retain_graph

        self.phases = ['train']
        if val_data_loader is not None:
            self.phases.append('validation')

        if test_data_loader is not None:
            self.phases.append('test')

        self.validation_only = validation_only
        if validation_only:
            self.phases = ['validation']
            print('Running in validation only mode.')

        self.data_loaders_ = {
            'train': data_loader,
            'validation': val_data_loader,
            'test': test_data_loader
        }

        if start_epoch > 0:
            self.start_ep = start_epoch + 1

        if not torch.cuda.is_available():
            warnings.warn('No GPU detected. Training can be slow.')

    def __call__(self, epochs_):
        """Train the model for a specified number of epochs

        Parameters
        ----------
        epochs_ : int
            how many epochs for training
        """
        self.printer.restart()

        do_stop = False

        if self.validation_only:
            num_batches = 0
        else:
            num_batches = len(self.data_loaders_['train'])

        # TODO: if resume, compute the learning rate beforehand
        if self.start_ep > 0:
            if self.batch_scheduler is not None:
                self._batch_schedule(
                    'train', self.start_ep, 0,
                    self.data_loaders_['train'].batch_size
                )
            if self.scheduler is not None:
                raise NotImplementedError

        print('STARTING TRAINING')
        for epoch in range(self.start_ep, epochs_):
            self.printer.learning_rate = get_lr(self.optimizer)

            for current_phase in self.phases:
                if current_phase == 'train':
                    self.train_interface.model.train()
                else:
                    self.train_interface.model.eval()

                self.time_logger.start(current_phase + '_load_data')
                for idx, sample in enumerate(self.data_loaders_[current_phase]):
                    self.time_logger.stop(current_phase + '_load_data')

                    self._batch_schedule(
                        current_phase, epoch, idx, num_batches
                    )

                    loss, metrics, counters, functions = self._iteration_step(
                        sample, current_phase)
                    self._update_printer(
                        epoch, loss, metrics, counters, functions, current_phase
                    )

                    if current_phase == 'train':
                        self._update_weights(loss)

                    self.time_logger.start(current_phase + '_load_data')
                    # ----------- end of phase ----------------------------

                self.time_logger.stop(
                    current_phase + '_load_data', do_log=False
                )

                # do certain actions depending on which phase we are in
                if self.early_stopping is not None and current_phase == 'validation':
                    do_stop = self.early_stopping(
                        self.printer.get_metrics(),
                        self.train_interface.model,
                        self.save_path,
                        self.save_state_dict
                    )
                self.printer.on_epoch_end()

                self._schedule(current_phase)
                self._save(epoch, epochs_, current_phase)

                # compute statistics on time values that are collected during
                # the upper for-loop
                if self.time_log_printer is not None:
                    self.time_log_printer.update(
                        torch.tensor([-1]), epoch, metrics=self.time_logger.get_data()
                    )
                    self.time_log_printer.on_epoch_end()
                    self.time_logger.restart()

            if do_stop:
                return
            # -------------------end of epoch -------------------------------

    def _iteration_step(self, sample, current_phase):
        """Compute loss and metrics

        Parameters
        ----------
        sample : anything provided by the data loader
            typically the sample x and the corresponding label
        current_phase : str
            training or validation

        Returns
        -------
        float, dict
            loss value that is used for gradient computation and a dictionary
            with metrics.
        """
        self.time_logger.start(current_phase + '_iteration_step')
        if current_phase == 'validation':
            with torch.no_grad():
                #loss, metrics, counters = self.train_interface.val_step(sample)
                output = self.train_interface.val_step(sample)
        elif current_phase == 'test':
            with torch.no_grad():
                output = self.train_interface.test_step(sample)

        else:
            #loss, metrics, counters = self.train_interface.train_step(sample)
            output = self.train_interface.train_step(sample)

        functions = None
        counters = None
        if len(output) == 2:
            loss, metrics = output[0], output[1]
        elif len(output) == 3:
            loss, metrics, counters = output[0], output[1], output[2]
        else:
            loss, metrics, counters = output[0], output[1], output[2]
            functions = output[3]

        self.time_logger.stop(current_phase + '_iteration_step')
        return loss, metrics, counters, functions

    def _update_weights(self, loss):
        """Compute gradient and apply backpropagation

        Parameters
        ----------
        loss : float
            error function the weight update is based on
        """
        self.time_logger.start('update_weights')

        self.optimizer.zero_grad()

        self.time_logger.start('loss_backward')
        loss.backward(retain_graph=self.retain_graph)
        self.time_logger.stop('loss_backward')

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.train_interface.model.parameters(), self.clip
            )

        self.time_logger.start('opt_step')
        self.optimizer.step()
        self.time_logger.stop('opt_step')

        self.time_logger.stop('update_weights')

    def _update_printer(self, epoch, loss, metrics, counters, functions, current_phase):
        """Pass the necessary values to the printer

        Parameters
        ----------
        epoch : int
            current epoch
        loss : float
            current loss value
        metrics : dict
        current_phase : str
            if the current phase is validation, all metrics/losses are renamed 
            to val_[name]

        """
        self.time_logger.start(current_phase + '_update_printer')
        if current_phase == 'train':
            self.printer.update(loss, epoch, metrics, counters, functions)
        else:
            prefix = {'validation': 'val_', 'test': 'test_'}[current_phase]
            if metrics is not None:
                metrics = {prefix + k: v for (k, v) in metrics.items()}
            if counters is not None:
                counters = {prefix + k: v for (k, v) in counters.items()}
            if functions is not None:
                functions = {prefix + k: v for (k, v) in functions.items()}

            self.printer.update(
                loss, epoch, metrics,
                counters, functions, loss_key=prefix + 'loss'
            )
        self.time_logger.stop(current_phase + '_update_printer')

        self.printer.print_conditional()

    def _schedule(self, current_phase):
        """update the scheduler after each epoch
        """
        if self.scheduler is not None:
            if current_phase == 'train':
                self.time_logger.start('schedule')
                self.scheduler.step()
                self.time_logger.stop('schedule')

    def _batch_schedule(self, current_phase, epoch, iteration, num_batches):
        """update the scheduler after each batch
        """
        if self.batch_scheduler is not None:
            if current_phase == 'train':
                self.time_logger.start('batch_schedule')
                self.batch_scheduler.step(epoch, iteration, num_batches)
                self.time_logger.stop('batch_schedule')

    def _save(self, epoch, epochs_, current_phase):
        """save the model to model path every 'save_steps' epochs.

        Parameters
        ----------
        epoch : int
            current epoch
        epochs_ : int
            number of epochs for entire training
        current_phase: str
            is this function called after training, val or testing? Only after
            validation, the model is saved.
        """

        # only save after validation
        if current_phase != 'validation':
            return

        self.time_logger.start('save')
        if self.do_save:
            if epoch == epochs_ - 1 or epoch % self.save_steps == 0:
                print(f'Saving {self.save_path}')
                if self.save_state_dict:
                    print('save as state dict')
                    to_save = self.train_interface.model.state_dict()

                    torch.save(
                        to_save,
                        self.save_path
                    )
                else:
                    torch.save(self.train_interface.model, self.save_path)
        self.time_logger.stop('save')
        print('logged save value')


def get_optimizer(opt_id, parameters, learning_rate, **kwargs):
    """ Simple getter function for a pytorch optimizer

    Parameters
    ----------
    opt_id : str
        which optimizer, e.g. SGD or Adam
    parameters : model.parameters
        pytorch variables that shall be updated, usually model.parameters()
        is passed
    learning_rate : float

    Returns
    -------
    pytorch optimizer

    Raises
    ------
    ValueError
        if unknown opt_id
    """
    if opt_id == 'SGD':
        optimizer = optim.SGD(parameters,
                              lr=learning_rate,
                              momentum=kwargs.get('momentum', .9),
                              weight_decay=kwargs.get('weight_decay', 0.),
                              nesterov=kwargs.get('nesterov', False)
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
    elif opt_id == 'AdaDelta':
        optimizer = optim.Adadelta(
            parameters,
            lr=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0.),
            rho=kwargs.get('rho', 0.9),
            eps=kwargs.get('eps', 1e-3)
        )
    else:
        raise ValueError(f'Unknown opt value: {opt_id}')

    return optimizer


def get_scheduler(lr_steps, epochs, optimizer, gamma=.1, fixed_steps=None):
    """returns a pytorch scheduler

    Parameters
    ----------
    lr_steps : int
        the learning rate is altered in 'lr_steps' uniformly steps
    epochs : int
        number of epochs for the entire training
    optimizer : pytorch optimizer
    gamma : float, optional
        the learning rate is multiplied by gamma, by default .1

    Returns
    -------
    pytorch scheduler
    """

    if fixed_steps is not None:
        assert lr_steps == 0, 'no lr_steps if fixed steps is used'
        # might be filled with strings, when coming from argparse
        fixed_steps = [int(x) for x in fixed_steps]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, fixed_steps,
            gamma=gamma
        )
        print(f'fixed rate scheduling at: {fixed_steps}')
        return scheduler

    if lr_steps < 1:
        return None

    assert lr_steps < epochs, f'Epochs must be greater than lr_steps but e:{epochs} < l:{lr_steps}'
    step_size = epochs // lr_steps
    print(f'Sched step size: {step_size}')

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size,
        gamma=gamma, last_epoch=-1
    )

    return scheduler


def set_device(device=None):
    """Use if you have multiple GPUs, but you only want to use a subset.
    Use the command nvidia-smi in the terminal for more information on your
    pc's gpu setup

    Parameters
    ----------
    device : int, optional
        masks all devices but 'device'. By default None, all devices are
        visible
    """
    if device is not None:
        if isinstance(device, list):
            device = ','.join([str(x) for x in device])
        else:
            device = str(device)
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        print(f'using device {device}')


def set_random_seed(seed):
    """Sets a seed for all training related random functions. The seed is only
    identical on the same machine.

    Parameters
    ----------
    seed : int

    """
    print(f'Setting seed: {seed}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cv2.setRNGSeed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    # for debugging purposes some random numbers are generated
    # output = {
    #    'seed': seed,
    #    'torch': torch.randn(1).item(),
    #    'cuda': torch.cuda.FloatTensor(1).normal_().item(),
    #    'numpy': float(np.random.randn(1)),
    #    'python': random.randint(0, 5000)
    # }
    # with open(os.path.join(options.folder_name, 'rand_num_test.json'), 'w') as file:
    #    json.dump(output, file)

    return _init_fn


def loss_verification(train_interface, data_loader, printer):
    """Run through one epoch and print the corresponding loss.
    When using cross-entropy, the usual loss should be -ln(num_classes). If
    not, there might be something wrong with your code.

    Parameters
    ----------
    train_interface : ITrainInterface
    data_loader : pytorch data_loader
    printer : Printer (pt_train_printer.py)
    """
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
    """Save the best model depending on a specified metric on the validation
    set.

    Returns
    -------
    EarlyStopping
    """

    def __init__(self, metric_key, get_max=True, epoch_thres=np.inf):
        """Constructor. You need to specify which metric should be observed,
        if the value is better when decreased or increased.
        For example:
        EarlyStopping('val_acc', get_max=True, epoch_thres=10)
        keeps track of the validation accuracy. If the current best validation
        accuracy (starting from -inf) is exceeded, this value is saved and 
        the model is saved.

        If after 10 epochs the best accuracy is not exceeded, the training
        is stopped.

        This object is used within the Training class.

        Parameters
        ----------
        metric_key : str
            Which metric is observed. Needs to be a metric that is present in
            the training_interface. val_[name] is also possible.
        get_max : bool, optional
            Save the model if the new observed metric is above the current best
            value (True) or below it (False). By default True.
        epoch_thres : int, optional
            if the model has not bin saved for 'epoch_thres' epochs,
            the training is stopped. By default np.inf, the model is trained 
            the full number of epochs.
        """
        self.key = metric_key
        self.get_max = get_max

        self.no_update_counter = 0.
        self.thres = epoch_thres

        if get_max:
            self.current_val = -np.inf
        else:
            self.current_val = +np.inf

    def __call__(self, metrics, model, save_path, save_state_dict):
        value = metrics[self.key]

        self.no_update_counter += 1
        if self.get_max:
            if value > self.current_val:
                self._update(value, model, save_path, save_state_dict)
        else:
            if value < self.current_val:
                self._update(value, model, save_path, save_state_dict)

        if self.no_update_counter > self.thres:
            return True
        else:
            return False

    def _update(self, value, model, save_path, save_state_dict):
        self.no_update_counter = 0
        self.current_val = value

        print(f'saving model: {save_path}')
        if save_state_dict:
            print('save as state dict')
            to_save = model.state_dict()
            torch.save(to_save, save_path)
        else:
            torch.save(model, save_path)


def get_printer(print_intervall, log_file=None):
    return Printer(print_intervall, log_file=log_file)


# taken from https://github.com/d-li14/mobilenetv2.pytorch/blob/master/imagenet.py

class BatchScheduler():
    def __init__(self, decay_type, optimizer, initial_learning_rate, warmup, num_epochs, gamma=.1):
        self.optimizer = optimizer
        self.lr = initial_learning_rate
        self.warmup = warmup
        self.num_epochs = num_epochs
        self.decay_type = decay_type
        self.gamma = gamma

    def step(self, epoch, iteration, num_iter):
        lr = self.optimizer.param_groups[0]['lr']

        warmup_epoch = 5 if self.warmup else 0
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = self.num_epochs * num_iter

        if self.decay_type == 'step':
            lr = self.lr * \
                (self.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
        elif self.decay_type == 'cos':
            lr = self.lr * \
                (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        elif self.decay_type == 'linear':
            lr = self.lr * (1 - (current_iter - warmup_iter) /
                            (max_iter - warmup_iter))
        else:
            raise ValueError('Unknown lr mode {}'.format(self.decay_type))

        if epoch < warmup_epoch:
            lr = self.lr * current_iter / warmup_iter

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class TimeLogger():
    def __init__(self, is_active):
        self.is_active = is_active
        self.data = dict()
        self.qu = dict()
        self.functions = {
            'mean': np.mean,
            'min': np.min,
            'max': np.max,
            'std': np.std,
            'median': np.median,
            'sum': np.sum
        }

    def restart(self):
        if not self.is_active:
            return
        self.data = dict()

    def start(self, key):
        if not self.is_active:
            return
        assert key not in self.qu.keys()
        self.qu[key] = time.time()

    def stop(self, key, do_log=True):
        if not self.is_active:
            return
        start_time = self.qu.pop(key)
        time_needed = time.time() - start_time
        if do_log:
            self._update(key, time_needed)

    def _update(self, key, value):
        assert self.is_active
        if key not in self.data.keys():
            self.data[key] = [value]
        else:
            self.data[key].append(value)

    def get_data(self):
        assert self.is_active

        out = dict()
        for key, values in self.data.items():
            values = np.array(values)
            for name, fcn in self.functions.items():
                tmp = float(fcn(values))
                out[key + '_' + name] = tmp

        return out
