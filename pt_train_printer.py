import json
import time


class Printer(object):
    def __init__(self, print_intervall, log_file=None):
        self.print_intervall = print_intervall
        self.log_file = log_file

        if self.log_file is not None:
            with open(self.log_file, 'w') as file:
                output_dict = dict()
                json.dump(output_dict, file)
        self.restart()

    def restart(self):
        self.loss = 0.0
        self.epoch = -1
        self.counter = 0.0
        self.learning_rate = -1.
        self.metrics = dict()
        self.start_time = time.time()
        self.time_needed = 0.
        self.loss_key = 'loss'

    def update(self, loss, epoch, metrics=None, loss_key='loss'):
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

    def print_conditional(self):
        if self.counter % self.print_intervall == 0:
            self.print()

    def print(self):
        print(self.get_out_str())

    def get_out_str(self):
        out_str = f'Ep: {self.epoch}, {self.loss_key}: {self.loss/self.counter:.5f}'
        for key, val in self.metrics.items():
            out_str += f' {key}: {val/self.counter:.3f}'
        out_str += f' lr: {self.learning_rate:.5f}'
        out_str += f' sec: {self.time_needed:.2f}'
        return out_str

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

        with open(self.log_file, 'w') as file:
            json.dump(output_dict, file)

    def _check_write(self, output_dict, key, value):
        if key not in output_dict.keys():
            output_dict[key] = [value]
        else:
            output_dict[key].append(value)
        return output_dict

    def on_epoch_end(self):
        self.print()
        if self.log_file is not None:
            self.write_to_log()
        self.restart()

    def get_metrics(self):
        assert self.counter > 0
        return {k: v / self.counter for (k, v) in self.metrics.items()}


if __name__ == "__main__":
    Printer(0, './printer_test.json')
