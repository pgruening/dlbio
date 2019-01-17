import copy
import warnings
from random import shuffle

import numpy as np

from DLBio import ds_keras_data_sequence
from DLBio.global_constants import DATA_ID_INDEX

DEFAULT_DEBUG = False


class DataSequence(ds_keras_data_sequence.DataSequence):
    def __init__(self, file_paths,
                 augmentation_functions,
                 batch_size,
                 ** kwargs
                 ):
        self.file_paths = copy.copy(file_paths)

        if not augmentation_functions.is_setup:
            warnings.warn('You did not run the setup method of the' +
                          ' augmentation list.')

        self.augmentation_functions = augmentation_functions
        # kwargs
        used_sample_id_list = kwargs.pop('used_sample_id_list', None)
        num_id_repeats = kwargs.pop('num_id_repeats', 1)
        return_ID = kwargs.pop('return_ID', False)
        shuffle = kwargs.pop('shuffle', True)
        self.DEBUG = kwargs.pop('DEBUG', DEFAULT_DEBUG)

        if kwargs:
            print('Warning: kwargs contains unused keys: {}'.format(
                kwargs.keys()))

        self.shuffle = shuffle
        self.batch_size = batch_size

        self.return_ID = return_ID

        self.num_id_repeats = num_id_repeats

        self.index_dict = dict()
        self._setup(used_sample_id_list)

    def _setup(self, used_sample_id_list):
        """Check all files if they contain the used samples.
        Otherwise they are discarded from the load list

        Parameters
        ----------
        used_sample_id_list : list of srt
            which samples are to be used by this generator

        """
        self.file_lengths = dict()
        self.len = 0

        files_to_remove = []
        for file_path in reversed(self.file_paths):
            data = np.load(file_path)

            index_list = self._get_index_list(data, used_sample_id_list)
            if not index_list:
                files_to_remove.append(file_path)
                continue

            self.data = data
            self.current_loaded_file = file_path
            self.index_list = index_list

            self.index_dict[file_path] = index_list

            self.file_lengths[file_path] = int(np.ceil(
                float(len(index_list))/float(self.batch_size)))

            self.len += self.file_lengths[file_path]

        for f_p in files_to_remove:
            self.file_paths.remove(f_p)

        self.has_labels = len(self.data[0]) >= 3

    def __getitem__(self, index):
        file_path, index_within_file = self._get_file_name(index)
        if self.current_loaded_file != file_path:
            self.data = np.load(file_path)
            self.index_list = self.index_dict[file_path]
            self.current_loaded_file = file_path

        return super(DataSequence, self).__getitem__(index_within_file)

    def _get_file_name(self, index):
        counter = 0
        for file_path in self.file_paths:
            if index < counter + self.file_lengths[file_path]:
                index_within_file = index - counter
                return file_path, index_within_file
            counter += self.file_lengths[file_path]

    def __len__(self):
        return self.len

    def _get_index_list(self, data, used_sample_id_list):
        if used_sample_id_list is None:
            index_list = [i for i in range(data.shape[0])]

        else:
            index_list = [i for i in range(data.shape[0])
                          if data[i][DATA_ID_INDEX] in used_sample_id_list
                          ]

        return index_list

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.file_paths)
            for index_list in self.index_dict.values():
                shuffle(index_list)
            super(DataSequence, self).on_epoch_end()

    def _make_index_list(self):
        raise NotImplementedError

    def print_augmentation_functions(self):
        raise NotImplementedError

    def print_ids(self):
        raise NotImplementedError

    def set_used_samples(self):
        raise NotImplementedError


def __debug():
    import DLBio.aug_generator_functions as aug
    from DLBio.aug_augmentation_list import AugmentationList
    import matplotlib.pyplot as plt

    file_paths = ['experiments/test_00.npy',
                  'experiments/test_01.npy',
                  'experiments/test_02.npy',
                  'experiments/test_03.npy',
                  'experiments/test_04.npy',
                  'experiments/test_05.npy',
                  'experiments/test_06.npy',
                  'experiments/test_07.npy']

    augmentations = AugmentationList([
        aug.Aug_GetFirstItemOfLabelList()
    ])

    # TODO: check ids
    cv_ids = ["K170302DB-6-Bild-8", "K170302DB-29-Bild-5",
              "K170302DB-18-Bild-13",
              "K170302DB-26-Bild-7", "K170302DB-17-Bild-13"]
    generator = DataSequence(file_paths, augmentations, 1,
                             used_sample_id_list=cv_ids, return_ID=True)
    for j in range(3):
        for i in range(len(generator)):
            batch = generator[i]
            print(i, len(batch[0]), batch[2])


if __name__ == '__main__':
    __debug()
