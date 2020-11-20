import time
import warnings

import torch
import torch.multiprocessing as mp

from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_lr


class InterfaceProcess():
    def __init__(self, train_interface, answer_qu):
        self.ti = train_interface
        self.ask_qu = mp.Queue()
        self.answer_qu = answer_qu

    def run(self):
        while True:
            time.sleep(2)
            if self.ask_qu.empty():
                continue

            tmp = self.ask_qu.get()
            cmd = tmp[0]
            print(f'found command {cmd}')
            if cmd == 'step':
                current_phase, sample, epoch = tmp[1], tmp[2], tmp[3]
                loss, metrics, counters, functions = _iteration_step(
                    self.ti, sample, current_phase)

                _update_printer(
                    self.ti, epoch, loss, metrics,
                    counters, functions, current_phase
                )

                if current_phase == 'train':
                    _update_weights(self.ti, loss)

                print('step done')
                self.answer_qu.put('step done')

            elif cmd == 'epoch_end':
                self.ti.printer.on_epoch_end()
                _schedule(self.ti, current_phase)

            elif cmd == 'set_printer_lr':
                self.ti.printer.learning_rate = get_lr(self.ti.optimizer)

            elif cmd == 'restart_printer':
                self.ti.printer.restart()

            elif cmd == 'set_models_to_train':
                self.ti.model.train()

            elif cmd == 'set_eval_model_in_models':
                self.ti.model.eval()


class IMultiModelTrainingInterface():
    def __init__(self, **kwargs):
        # these values need to be set
        self.model = kwargs.get('model')
        self.optimizer = kwargs.get('optimizer')

        save_steps = kwargs.get('save_steps', -1)
        save_path = kwargs.get('save_path', None)
        save_state_dict = kwargs.get('save_state_dict', False)

        assert isinstance(save_steps, int)
        if save_steps > 0:
            assert save_path is not None
        self.do_save = save_steps > 0 and save_path is not None
        self.save_steps = save_steps
        self.save_path = save_path
        self.save_state_dict = save_state_dict
        print(self.save_state_dict)

        self.clip = kwargs.get('clip', None)
        self.retain_graph = kwargs.get('retain_graph', False)

        self.printer = kwargs.get('printer', Printer(100, None))
        self.scheduler = kwargs.get('scheduler', None)

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


class MultiModelTraining():
    def __init__(self, data_loader, train_interfaces, device, val_data_loader=None, test_data_loader=None, validation_only=False):
        self.data_loader = data_loader
        self.qu = mp.Queue()
        self.train_interfaces = [
            InterfaceProcess(x, self.qu) for x in train_interfaces
        ]

        self.phases = ['train']
        if val_data_loader is not None:
            self.phases.append('validation')

        if test_data_loader is not None:
            self.phases.append('test')

        if validation_only:
            self.phases = ['validation']
            print('Running in validation only mode.')

        self.data_loaders_ = {
            'train': data_loader,
            'validation': val_data_loader,
            'test': test_data_loader
        }

        self.d = device

        if not torch.cuda.is_available():
            warnings.warn('No GPU detected. Training can be slow.')

    def __call__(self, epochs_):
        try:
            processes = []
            for ti in self.train_interfaces:
                tmp = mp.Process(target=ti.run)
                tmp.start()
                processes.append(tmp)

            self.__run(epochs_)
        except:
            for tmp in processes():
                tmp.terminate()

    def __run(self, epochs_):

        self._restart_printer()

        print('STARTING TRAINING')

        for epoch in range(epochs_):
            self._set_printer_lr()

            for current_phase in self.phases:
                if current_phase == 'train':
                    self._set_models_to_train()
                else:
                    self._set_eval_model_in_models()

                for sample in self.data_loaders_[current_phase]:
                    sample = [x.to(self.d) for x in sample]
                    for ti in self.train_interfaces:
                        ti.ask_qu.put(['step', current_phase, sample, epoch])
                        # check for train interfaces that can be removed

                    is_done_ctr = 0
                    while is_done_ctr < len(self.train_interfaces):
                        if not self.qu.empty():
                            tmp = self.qu.get()
                            if tmp == 'step done':
                                is_done_ctr += 1
                        print(is_done_ctr)
                        time.sleep(2.)

                for ti in self.train_interfaces:
                    ti.ask_qu.put(['on_epoch_end'])

    def _set_printer_lr(self):
        for ti in self.train_interfaces:
            # ti.printer.learning_rate = get_lr(ti.optimizer)
            ti.ask_qu.put(['set_printer_lr'])

    def _restart_printer(self):
        for ti in self.train_interfaces:
            ti.ask_qu.put(['restart_printer'])
#            ti.printer.restart()

    def _set_models_to_train(self):
        for ti in self.train_interfaces:
            ti.ask_qu.put(['set_models_to_train'])
#          ti.model.train()

    def _set_eval_model_in_models(self):
        for ti in self.train_interfaces:
            ti.ask_qu.put(['set_eval_model_in_models'])
 #           ti.model.eval()


def _iteration_step(train_interface, sample, current_phase):
    if current_phase == 'validation':
        with torch.no_grad():
            # loss, metrics, counters = self.train_interface.val_step(sample)
            output = train_interface.val_step(sample)
    elif current_phase == 'test':
        with torch.no_grad():
            output = train_interface.test_step(sample)

    else:
        # loss, metrics, counters = self.train_interface.train_step(sample)
        output = train_interface.train_step(sample)

    functions = None
    counters = None
    if len(output) == 2:
        loss, metrics = output[0], output[1]
    elif len(output) == 3:
        loss, metrics, counters = output[0], output[1], output[2]
    else:
        loss, metrics, counters = output[0], output[1], output[2]
        functions = output[3]

    return loss, metrics, counters, functions


def _update_printer(train_interface, epoch, loss, metrics, counters, functions, current_phase):
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
        train_interface.printer.update(
            loss, epoch, metrics, counters, functions)
    else:
        prefix = {'validation': 'val_', 'test': 'test_'}[current_phase]
        if metrics is not None:
            metrics = {prefix + k: v for (k, v) in metrics.items()}
        if counters is not None:
            counters = {prefix + k: v for (k, v) in counters.items()}
        if functions is not None:
            functions = {prefix + k: v for (k, v) in functions.items()}

        train_interface.printer.update(
            loss, epoch, metrics,
            counters, functions, loss_key=prefix + 'loss'
        )

    train_interface.printer.print_conditional()


def _update_weights(train_interface, loss):
    """Compute gradient and apply backpropagation

    Parameters
    ----------
    loss : float
        error function the weight update is based on
    """
    train_interface.optimizer.zero_grad()
    loss.backward(retain_graph=train_interface.retain_graph)

    if train_interface.clip is not None:
        torch.nn.utils.clip_grad_norm_(
            train_interface.train_interface.model.parameters(),
            train_interface.clip
        )

    train_interface.optimizer.step()


def _schedule(train_interface, current_phase):
    """update the scheduler after each epoch
    """
    if train_interface.scheduler is not None:
        if current_phase == 'train':
            train_interface.scheduler.step()


def _save(train_interface, epoch, epochs_):
    """save the model to model path every 'save_steps' epochs.

    Parameters
    ----------
    epoch : int
        current epoch
    epochs_ : int
        number of epochs for entire training
    """
    if train_interface.do_save:
        if epoch == epochs_ - 1 or epoch % train_interface.save_steps == 0:
            print(f'Saving {train_interface.save_path}')
            if train_interface.save_state_dict:
                print('save as state dict')
                to_save = train_interface.train_interface.model.state_dict()

                torch.save(
                    to_save,
                    train_interface.save_path
                )
            else:
                torch.save(train_interface.train_interface.model,
                           train_interface.save_path)
