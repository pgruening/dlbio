import json
import time
from os.path import isfile


class IPrinterFcn():
    """
    Interface for a function that the Printer uses. It needs to
    implement three methods: update, restart, and __call__ to be used by the
    Printer. Printer metrics represent the average (over an epoch) of values
    already averaged over a mini-batch. Computing the mean of
    already averaged values may not represent the metric truthfully.
    E.g.:
    sum .5*(.33*[a, b, c] + .33*[d, e, f]) != sum .16 * [a, b, c, d, e, f].

    An IPrinterFcn can store the values [a, b, c] and [d, e, f] in an
    update-call after each batch-computation. Then return
    sum .16 * [a, b, c, d, e, f] in the __call__ method. In restart, the memory
    of the function is flushed.

    Note that in a TrainingInterface, functions are usually passed as a
    dictionary to the Printer.

    """

    def update(self):
        """
        This function ought to be called in a train-interface step.
        Use it, e.g., to store values in the class object for later
        processing in __call__
        """
        raise NotImplementedError()

    def __call__(self):
        """
        Important:
        calling this method MUST NOT change the state of the object:
        res1 = IPrinterFcn()
        res2 = IPrinterFcn()
        assert res1 == res2

        res1 and res2 must be equal. Only restart and update should change
        the state of the class object!

        This function is called by the printer in 
        get_out_str -> each time the printer prints a line
        write_to_log -> at the end of an epoch 
        get_metrics -> is called during early stopping

        Returns one metric value to the printer
        """
        raise NotImplementedError()

    def restart(self):
        """
        This function is called by the printer in 'restart'.
        printer.restart() is called by printer.on_epoch_end() which is called
        during pt_training on the end of each epoch.
        If you save data in the class object, this is the function
        to remove them after an epoch.
        """
        raise NotImplementedError()


class Printer(object):
    def __init__(self, print_intervall, log_file=None, dont_print_list=None, resume=False):
        """A printer is usually stored in a TrainingInterface. During a
        pt_training process, it keeps the loss, learning rate, and other
        metrics that describe the training run. It prints intermediate results
        to the terminal. Furthermore, if a log_file string is specified,
        it writes its values to this log file after each epoch.

        Parameters
        ----------
        print_intervall : int
            each print_intervall batches, the printer writes something to the
            terminal. See the print_conditional function. If the value is -1,
            the printer never writes intermediate results.
        log_file : str, optional
            Path to a json file. On each epoch's end, the printer writes its
            results to this file. By default None, which means no file is
            written.
        dont_print_list : list of str, optional
            specify keys that are not printed to the terminal. Note however,
            that these values will be written to the log-file. By default None:
            all keys are written to the terminal
        resume : bool, optional
            If a log file specified in 'log_file' already exists, it is
            overwritten entirely. This happens when resume is set to False.
            If you are resuming a training process, you'd instead want to start
            with the already existing log-file and append new values to it.
            This happens when resume is set to True. By default False.
            The printer is usually initialized in a 'run_training.py' module.
            There, you'll need to make sure that the resume flag is set
            properly.
        """
        self.print_intervall = print_intervall
        self.log_file = log_file
        self.dont_print = dont_print_list

        # overwrite old log file, if it exists.
        if self.log_file is not None and not resume or not isfile(self.log_file):
            print(f'writing new log-file: {self.log_file}')
            with open(self.log_file, 'w') as file:
                output_dict = dict()
                json.dump(output_dict, file)

        # In the 'update'-call, this attribute is overwritten with
        # a dictionary. In restart, it is set back to 'None'.
        self.functions = None
        self.restart()

    def on_epoch_end(self):
        if self.print_intervall != -1:
            self.print()

        if self.log_file is not None:
            self.write_to_log()
        self.restart()

    def restart(self):
        if self.functions is not None:
            for fcn in self.functions.values():
                fcn.restart()

        # is a counter that is averaged by self.counter (= number of batches)
        # when printed
        self.loss = 0.0
        self.counter = 0.0

        self.epoch = -1
        self.learning_rate = -1.

        self.metrics = dict()
        self.counters = dict()
        # time needed measures the processing time for one epoch.
        self.time_needed = 0.
        self.start_time = time.time()
        # in different training phases (train, val, test), this value is
        # changed to loss, val_loss, and test_loss
        # (see pt_training, _update_printer)
        self.loss_key = 'loss'
        self.functions = None

    def update(self, loss, epoch, metrics=None, counters=None, functions=None, loss_key='loss'):
        """ Save loss values, learning rates, and other metrics.
        This function is called in pt_training's _update_printer().

        Metrics and counters make a lot of sense in tasks with many label
        values, for example, segmentation. Functions, on the other hand, work
        well on single label tasks, such as classification, or quality 
        assessment.

        Parameters
        ----------
        loss : float
            the loss function that is used to train the network
        epoch : int
            current training epoch
        metrics : dictionary of float, optional
            metric values are supposed to be averaged over mini-batches. Each
            metric value is used like a counter. When printed, each value is
            divided by self.counter. Thus, the mean of the average is reported.
            By default None, nothing is computed.
        counters : dictionary of float or int, optional
            Counters have the same behavior as metrics. However, during
            printing, the values are not normalized. The idea is to compute
            normalizations after the entire training.
            By default None, nothing is computed.
        functions : dictionary of IPrinterFcns, optional
            functions are just written to the self.functions attribute. They
            are called during printing operations.
            By default None, nothing is computed.
        loss_key : str, optional
            is changed depending on the training phase:
            loss, val_loss, and test_loss. By default 'loss'.
        """
        self.loss_key = loss_key
        self.loss += loss.item()
        self.counter += 1.0
        self.epoch = epoch

        self.time_needed += time.time() - self.start_time
        self.start_time = time.time()

        if metrics is not None:
            for key, val in metrics.items():
                if key not in self.metrics.keys():
                    self.metrics[key] = val
                else:
                    self.metrics[key] += val

        if counters is not None:
            for key, val in counters.items():
                if key not in self.counters.keys():
                    self.counters[key] = val
                else:
                    self.counters[key] += val

        if functions is not None:
            self.functions = functions

    def print_conditional(self):
        if self.print_intervall == -1:
            return
        if self.counter % self.print_intervall == 0:
            self.print()

    def print(self):
        print(self.get_out_str())

    def get_out_str(self):
        out_str = f'Ep: {self.epoch}, {self.loss_key}: {self.loss/self.counter:.5f}'
        for key, val in self.metrics.items():
            if self.do_print(key):
                out_str += f' {key}: {val/self.counter:.3f}'
        for key, val in self.counters.items():
            if self.do_print(key):
                out_str += f' {key}: {val:.3f}'

        if self.functions is not None:
            for key, fcn in self.functions.items():
                if self.do_print(key):
                    out_str += f' {key}: {fcn():.3f}'

        out_str += f' lr: {self.learning_rate:.5f}'
        out_str += f' sec: {self.time_needed:.2f}'
        return out_str

    def do_print(self, key):
        if self.dont_print is None:
            return True

        if 'val_' in key:
            key = key.split('val_')[1]

        if 'test_' in key:
            key = key.split('test_')[1]

        return key not in self.dont_print

    def write_to_log(self):
        # NOTE: pt_tensor values cannot be written to a json file
        with open(self.log_file, 'r') as file:
            output_dict = json.load(file)

        output_dict = self._check_write(output_dict, 'epoch', self.epoch)
        output_dict = self._check_write(
            output_dict, self.loss_key, self.loss / self.counter)

        output_dict = self._check_write(
            output_dict, 'sec', self.time_needed)

        output_dict = self._check_write(
            output_dict, 'lr', self.learning_rate)

        for key, val in self.metrics.items():
            output_dict = self._check_write(
                output_dict, key, val / self.counter)

        for key, val in self.counters.items():
            output_dict = self._check_write(
                output_dict, key, val)

        if self.functions is not None:
            for key, fcn in self.functions.items():
                output_dict = self._check_write(output_dict, key, fcn())

        with open(self.log_file, 'w') as file:
            json.dump(output_dict, file)

    def _check_write(self, output_dict, key, value):
        if key not in output_dict.keys():
            output_dict[key] = [value]
        else:
            output_dict[key].append(value)
        return output_dict

    def get_metrics(self):
        out = dict()
        if self.metrics is not None:
            out.update({k: v / self.counter for (k, v)
                        in self.metrics.items()})
        if self.counters is not None:
            out.update({k: v for (k, v) in self.counters.items()})
        if self.functions is not None:
            out.update({k: v() for (k, v) in self.functions.items()})
        return out


if __name__ == "__main__":
    Printer(0, './printer_test.json')
