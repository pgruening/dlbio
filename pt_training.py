import argparse
import os
import random
import warnings

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
            retain_graph=False, val_data_loader=None, early_stopping=None
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

        Returns
        -------
        Training object
        """
        self.optimizer = optimizer
        self.data_loader = data_loader

        assert issubclass(train_interface.__class__, ITrainInterface)
        self.train_interface = train_interface

        self.scheduler = scheduler
        self.early_stopping = early_stopping

        if printer is None:
            self.printer = Printer(100, None)
        else:
            self.printer = printer

        assert isinstance(save_steps, int)
        if save_steps > 0:
            assert save_path is not None

        self.do_save = save_steps > 0 and save_path is not None
        self.save_steps = save_steps
        self.save_path = save_path

        self.clip = clip
        self.retain_graph = retain_graph

        self.phases = ['train']
        if val_data_loader is not None:
            self.phases.append('validation')

        self.data_loaders_ = {'train': data_loader,
                              'validation': val_data_loader}

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
        if current_phase == 'validation':
            with torch.no_grad():
                loss, metrics = self.train_interface.val_step(sample)
        else:
            loss, metrics = self.train_interface.train_step(sample)
        return loss, metrics

    def _update_weights(self, loss):
        """Compute gradient and apply backpropagation

        Parameters
        ----------
        loss : float
            error function the weight update is based on
        """
        self.optimizer.zero_grad()
        loss.backward(retain_graph=self.retain_graph)

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.train_interface.model.parameters(), self.clip
            )

        self.optimizer.step()

    def _update_printer(self, epoch, loss, metrics, current_phase):
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
        if current_phase == 'train':
            self.printer.update(loss, epoch, metrics)
        else:
            if metrics is not None:
                metrics = {'val_' + k: v for (k, v) in metrics.items()}
            self.printer.update(loss, epoch, metrics, loss_key='val_loss')

        self.printer.print_conditional()

    def _schedule(self, current_phase):
        """update the scheduler after each epoch
        """
        if self.scheduler is not None:
            if current_phase == 'train':
                self.scheduler.step()

    def _save(self, epoch, epochs_):
        """save the model to model path every 'save_steps' epochs.

        Parameters
        ----------
        epoch : int
            current epoch
        epochs_ : int
            number of epochs for entire training
        """
        if self.do_save:
            if epoch == epochs_ - 1 or epoch % self.save_steps == 0:
                print(f'Saving {self.save_path}')
                torch.save(self.train_interface.model, self.save_path)


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


def get_scheduler(lr_steps, epochs, optimizer, gamma=.1):
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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
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

    def __call__(self, metrics, model, save_path):
        value = metrics[self.key]

        self.no_update_counter += 1
        if self.get_max:
            if value > self.current_val:
                self._update(value, model, save_path)
        else:
            if value < self.current_val:
                self._update(value, model, save_path)

        if self.no_update_counter > self.thres:
            return True
        else:
            return False

    def _update(self, value, model, save_path):
        self.no_update_counter = 0
        self.current_val = value
        torch.save(model, save_path)
        print(f'saving model: {save_path}')


def get_printer(print_intervall, log_file=None):
    return Printer(print_intervall, log_file=log_file)
