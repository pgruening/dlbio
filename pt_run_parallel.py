import random
import subprocess
import time
from multiprocessing import Process

NUM_TRIES = 1
AVAILABLE_GPUS = [0, 1, 2, 3]


def run(param_generator, make_object,
        available_gpus=AVAILABLE_GPUS, num_tries=NUM_TRIES,
        shuffle_params=True
        ):
    """Run a number of processes in parallel, while keeping all GPUs
    busy

    Parameters
    ----------
    param_generator : generator of dictionaries
        generates dictionaries with parameters for the process
    make_object : function 
        function that returns a class object which is a training process
        process should implement __call__, set_timer and have a __name__ and
        start_time property.
    """
    train_processes_ = []

    current_available_gpus = available_gpus

    for kwargs in param_generator:
        for try_num in range(num_tries):
            train_process = make_object(try_num, **kwargs)
            train_processes_.append(train_process)

    if shuffle_params:
        random.shuffle(train_processes_)

    active_processes_ = []
    while train_processes_ or active_processes_:
        if current_available_gpus and train_processes_:
            next_gpu = current_available_gpus.pop()

            train_process = train_processes_.pop()
            train_process.device = next_gpu

            p = Process(target=train_process)
            train_process.set_timer()
            p.start()

            active_processes_.append((p, train_process))

            print(f'starting process {train_process.__name__}')
            print(f'available gpus: {current_available_gpus}')

        time.sleep(5.)
        for (p, train_process) in active_processes_:
            if not p.is_alive():
                new_gpu = train_process.device
                current_available_gpus.append(new_gpu)

                active_processes_.remove((p, train_process))
                minutes_needed = (time.time() - train_process.start_time) / 60.

                print(f'process {train_process.__name__} is done.')
                print(f'minutes needed: {minutes_needed}')
                print(f'available gpus: {current_available_gpus}')


class MakeObject():
    """us me e.g. like this:
    def run():
        make_object = pt_run_parallel.MakeObject(TrainingProcess)
        pt_run_parallel.run(param_generator(), make_object,
                            available_gpus=AVAILABLE_GPUS
                            )
    """

    def __init__(self, TrainingProcess):
        self.TrainingProcess = TrainingProcess

    def __call__(self, try_num, **kwargs):
        # try num is not really used anymore...
        return self.TrainingProcess(**kwargs)


class ITrainingProcess():
    def __init__(self, **kwargs):
        # NOTE: run init with kwargs, save kwargs as attribute
        # run subprocess.call with self.kwargs in __call__
        self.start_time = -1
        self.device = -1

        self.__name__ = 'Give me a name!'
        self.module_name = 'some_name.py'
        self.kwargs = dict()

    def __call__(self):
        call_str = ['python', self.module_name]
        for key, value in self.kwargs.items():
            call_str += [f'--{key}']
            if value is not None:
                if isinstance(value, list):
                    call_str += [f'{x}' for x in value]
                else:
                    call_str += [f'{value}']

        # NOTE: make sure the called subprocess has this property
        if self.device is not None:
            call_str += ['--device', str(self.device)]

        print(call_str)
        subprocess.call(call_str)

    def set_timer(self):
        self.start_time = time.time()
