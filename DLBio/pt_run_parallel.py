import random
import subprocess
import time
from multiprocessing import Process
from os.path import splitext

import numpy as np
from tqdm import tqdm

from .pytorch_helpers import get_free_gpu_memory, get_free_gpus

AVAILABLE_GPUS = [0, 1, 2, 3]
GPU_MAN_THRES = 60.  # 60 seconds to block the gpu memory


def run(param_generator, make_object,
        available_gpus=AVAILABLE_GPUS,
        shuffle_params=True,
        do_not_check_free_gpus=False
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
    do_not_check_free_gpus: bool
        do not check if devices are currently used, default is False.
    """
    train_processes_ = []
    current_available_gpus = check_for_free_gpus(
        available_gpus, do_not_check=do_not_check_free_gpus, verbose=True)

    for kwargs in param_generator:
        train_process = make_object(0, **kwargs)
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

                current_available_gpus = check_for_free_gpus(
                    available_gpus, do_not_check=do_not_check_free_gpus)


def run_bin_packing(param_generator, make_object,
                    available_gpus=AVAILABLE_GPUS,
                    log_file=None,
                    max_num_processes=3,
                    shuffle_params=False,
                    setup_time=0.,
                    time_threshold=GPU_MAN_THRES
                    ):

    logger = TrainingLogger(log_file)
    gpu_manager = GPUManager(setup_time=setup_time,
                             time_threshold=time_threshold
                             )
    gpu_process_count = dict()
    train_processes_ = []

    print('Creating parameter list...')
    for kwargs in tqdm(param_generator):
        train_process = make_object(0, assert_mem_info=True, **kwargs)
        train_processes_.append(train_process)
    print('... done.')

    if shuffle_params:
        random.shuffle(train_processes_)

    # sort each process by memory usage (descending),
    # note shuffle only makes sense when the memory usage is equal for most
    # params
    train_processes_ = sorted(
        train_processes_, key=lambda x: x.mem_used, reverse=True
    )
    logger.log_processes(train_processes_)

    active_processes_ = []
    while train_processes_ or active_processes_:
        # greedy add processes to gpu
        for p_id, train_process in enumerate(train_processes_):
            for gpu_idx, free_memory in gpu_manager.items():
                if gpu_idx not in available_gpus:
                    continue

                if train_process.mem_used < free_memory:
                    if gpu_idx not in gpu_process_count.keys():
                        gpu_process_count[gpu_idx] = 0

                    # do not exceed to maximum number of process on one gpu
                    if gpu_process_count[gpu_idx] + 1 > max_num_processes:
                        continue
                    else:
                        gpu_process_count[gpu_idx] += 1

                    # block the amount of memory used for some time
                    gpu_manager.block_gpu(gpu_idx, train_process.mem_used)
                    # start process on free gpu
                    train_processes_.remove(train_process)
                    train_process.device = gpu_idx

                    p = Process(target=train_process)
                    train_process.set_timer()
                    p.start()

                    active_processes_.append((p, train_process, p_id, gpu_idx))

                    print(f'starting process {train_process.__name__}')
                    logger.log_start(
                        p_id, train_process.mem_used, gpu_idx, free_memory
                    )
                    break

        time.sleep(5.)

        # manage active processes
        for (p, train_process, p_id, gpu_idx) in active_processes_:
            if not p.is_alive():
                active_processes_.remove((p, train_process, p_id, gpu_idx))

                gpu_process_count[gpu_idx] -= 1
                assert gpu_process_count[gpu_idx] >= 0

                minutes_needed = (time.time() - train_process.start_time) / 60.

                print(f'process {train_process.__name__} is done.')
                print(f'minutes needed: {minutes_needed}')
                logger.log_end(p_id, minutes_needed)


def check_for_free_gpus(available_gpus, verbose=False, do_not_check=False):
    if do_not_check:
        return available_gpus

    free_gpus = get_free_gpus()
    current_available_gpus = list(
        set(available_gpus).intersection(set(free_gpus))
    )
    if verbose:
        print(f'Available GPUs: {current_available_gpus}')

    return current_available_gpus


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

    def __call__(self, try_num, assert_mem_info=False, **kwargs):
        if assert_mem_info:
            assert kwargs['mem_used'] is not None

        if 'mem_used' not in kwargs.keys():
            mem_used = None
        else:
            mem_used = kwargs.pop('mem_used')

        # most TrainingProcesses do not have mem_used as an attribute.
        # To be backwards compatible, mem_used is only set if it is an
        # attribute in the Class
        tp = self.TrainingProcess(**kwargs)
        if hasattr(tp, 'mem_used'):
            tp.mem_used = mem_used

        return tp


class ITrainingProcess():
    def __init__(self, **kwargs):
        # NOTE: run init with kwargs, save kwargs as attribute
        # run subprocess.call with self.kwargs in __call__
        self.start_time = -1
        self.device = -1
        self.mem_used = None

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


class GPUManager():
    def __init__(self, *, time_threshold, setup_time=0.):
        self.gpus = dict()
        self.thres = time_threshold
        self.setup_time = setup_time
        gpu_mem = get_free_gpu_memory()

        for gpu_idx, free_memory in gpu_mem.items():
            # start with all gpus unblocked
            self.gpus[gpu_idx] = {
                'free_mem': free_memory, 'timer': time.time() - 2 * self.thres
            }

    def items(self):
        for gpu_idx, gpu in self.gpus.items():
            yield gpu_idx, self._get_mem(gpu_idx)

    def _get_mem(self, gpu_idx):
        # read values from nvidia-smi
        smi_gpu_free_mem = get_free_gpu_memory()

        gpu = self.gpus[gpu_idx]

        # estimate time since the gpu was blocked
        t = time.time() - gpu['timer']

        if t < self.setup_time:
            # gpu is blocked entirely during setup time
            return 0.

        if t > self.thres:
            # gpu unblocked, use actual memory value
            return smi_gpu_free_mem[gpu_idx]
        else:
            # gpu is still blocked, return estimated memory value
            return gpu['free_mem']

    def block_gpu(self, idx, expected_mem_usage):
        # block gpu for a time to ensure that no new processes are added.
        # During setup for a training process, nvidia-smi does not show the
        # memory usage that is about to happen. The model needs to be loaded
        # first, etc...

        free_gpu_mem = self._get_mem(idx)

        self.gpus[idx]['free_mem'] = free_gpu_mem - expected_mem_usage
        self.gpus[idx]['timer'] = time.time()


class TrainingLogger():
    def __init__(self, path):
        self.path = path
        if self.path is not None:
            assert splitext(self.path)[-1] == '.txt'
            with open(self.path, 'w') as file:
                file.write('Starting Training \n')

    def log_processes(self, train_processes_):
        if self.path is None:
            return

        mem_values = np.array([x.mem_used for x in train_processes_])
        with open(self.path, 'a') as file:
            file.write(f'Found {mem_values.shape[0]} processes. \n')
            file.write(f'Mean exp. memory usage {mem_values.mean()} \n')
            file.write(f'Max exp. memory usage {mem_values[0]} \n')
            file.write(f'Min exp. memory usage {mem_values[-1]} \n')

    def log_start(self, p_id, mem_used, gpu_idx, free_memory):
        if self.path is None:
            return

        with open(self.path, 'a') as file:
            file.write(
                f'Add process {p_id} with exp. mem usage {mem_used} to GPU {gpu_idx} with free memory {free_memory} \n'
            )

    def log_end(self, p_id, minutes_needed):
        if self.path is None:
            return

        with open(self.path, 'a') as file:
            file.write(
                f'Process {p_id} stopped after {minutes_needed} minutes. \n'
            )
