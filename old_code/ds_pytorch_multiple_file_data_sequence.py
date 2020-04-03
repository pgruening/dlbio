import copy
import warnings

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch

from .ds_Igenerator import IGenerator
from .global_constants import DATA_ID_INDEX, DATA_IMAGE_INDEX, DATA_LABEL_INDEX
import torchvision.transforms as transforms
from .ds_pytorch_data_sequence import PyTorchDataset

import random
from collections import OrderedDict


class DataSequence(IGenerator):
    def __init__(self,
                 file_paths,
                 augmentation_functions,
                 batch_size,
                 used_sample_id_list=None,
                 num_id_repeats=1,
                 return_ID=False,
                 shuffle=True,
                 DEBUG=False,
                 num_workers=4,
                 to_pytorch_tensor=True
                 ):

        if DEBUG:
            num_workers = 0

        self.dataset = PyTorchMultiFileDataset(file_paths,
                                               augmentation_functions,
                                               used_sample_id_list,
                                               num_id_repeats,
                                               return_ID,
                                               batch_size,
                                               DEBUG,
                                               to_pytorch_tensor)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=MultiFileSampler(self.dataset.get_num_indeces_list())
        )

    # def __getitem__(self, index):
    #    return self.data_loader.__getitem__(index)

    def __iter__(self):
        return self.data_loader.__iter__()

    def __len__(self):
        return self.data_loader.__len__()

    def is_match(self, index, used_sample_id_list):
        return self.data[index][DATA_ID_INDEX] in used_sample_id_list


class PyTorchMultiFileDataset(PyTorchDataset):
    def __init__(self,
                 file_paths,
                 augmentation_functions,
                 used_sample_id_list,
                 num_id_repeats,
                 return_ID,
                 batch_size,
                 DEBUG,
                 to_pytorch_tensor,
                 image_transform=transforms.ToTensor(),
                 label_transform=transforms.ToTensor()
                 ):

        self.batch_size = batch_size

        self.data_id_index = DATA_ID_INDEX

        self.augmentation_functions = augmentation_functions

        self.return_ID = return_ID

        self.id_list = copy.copy(used_sample_id_list)

        self.DEBUG = DEBUG

        self.to_pytorch_tensor = to_pytorch_tensor
        self.image_to_tensor = image_transform
        self.label_to_tensor = label_transform

        self.file_paths = copy.copy(file_paths)
        self.index_dict = dict()

        self._setup(used_sample_id_list)

    def __getitem__(self, index):
        """Find the file that contains the given index and load it.
        With this file as self.data run the get_item code of the parent class.
        Parameters
        ----------
        index : int
            sample index

        Returns
        -------
        batch
            return the data_generators batch
        """
        #print('counter index')
        # print(index)
        file_path, index_within_file = self._get_file_name(index)
        if self.current_loaded_file != file_path:
            # print('Reloading')
            self._set_new_file(file_path)

        return PyTorchDataset.__getitem__(self, index_within_file)

    def _get_index_list(self, data, used_sample_id_list):
        if used_sample_id_list is None:
            index_list = [i for i in range(data.shape[0])]

        else:
            index_list = self.index_list = [i for i in range(
                data.shape[0]) if self.is_match(i, data, used_sample_id_list)]

        return index_list

    def get_num_indeces_list(self):
        return list(self.num_indeces.values())

    def _setup(self, used_sample_id_list):
        """Check all files if they contain the used samples.
        Otherwise they are discarded from the load list

        Variables that are setup:
        Does setup the index dict that keeps track of which indeces per
        file are matching the cv_ids
        index_dict = {file_path: ["indeces that match cv_ids"]}
        file_lengths = {file_path: "num batches for one run"}
        len = sum of all batches per file


        Parameters
        ----------
        used_sample_id_list : list of srt
            which samples are to be used by this generator

        """
        self.file_lengths = OrderedDict()
        self.num_indeces = OrderedDict()
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
            self.num_indeces[file_path] = len(index_list)

            self.len += self.file_lengths[file_path]

        for f_p in files_to_remove:
            self.file_paths.remove(f_p)

        # load the file that is on the first position of the ordered dict
        file_path = list(self.num_indeces.keys())[0]
        self._set_new_file(file_path)

        self.has_labels = len(self.data[0]) >= 3

    def _set_new_file(self, file_path):
        self.current_loaded_file = file_path
        self.data = np.load(file_path)
        self.index_list = self.index_dict[file_path]

    def __len__(self):
        return self.len
#        return len(self.index_list)

    def is_match(self, index, data, used_sample_id_list):
        return data[index][DATA_ID_INDEX] in used_sample_id_list

    def _get_file_name(self, index):
        """Returns the file path which contains the index. Furthermore,
        the index within the actual file is computed.

        Parameters
        ----------
        index : int
            sample index

        Returns
        -------
        tuple: (str, int)
            file_path and index within that file
        """
        counter = 0
        for file_path in self.num_indeces.keys():
            # self.file_lengths[file_path]:
            if index < counter + self.num_indeces[file_path]:
                index_within_file = index - counter
                return file_path, index_within_file
#            counter += self.file_lengths[file_path]
            counter += self.num_indeces[file_path]

    # def to_tensor(self, pic):
    #    return torch.from_numpy(np.flip(pic.transpose((2, 0, 1)), axis=0).copy())
# TODO add dataloader


class MultiFileSampler(Sampler):
    def __init__(self, file_lenghts):
        print('Initialized Sampler')
        self.file_lenghts = file_lenghts

        self.len = 0
        for x in file_lenghts:
            self.len += x

        # NOTE: I don't think you are allowed to shuffle those data!
        # random.shuffle(file_lenghts)
        self._start()

    def _start(self):
        # create an index list from 0 to num_indeces of the first file
        self.global_index = 0
        self.index_in_list = 0
        self.list_index = 0

        # output is the index within the list and all lists that were already
        # seen
        self.offset = 0

        self._setup_new_list()

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def _setup_new_list(self):
        # we know in the new file are n items. Compute random indeces for which
        # item is grabbed next
        new_list = list(range(self.file_lenghts[self.list_index]))
        random.shuffle(new_list)
        self.current_list = new_list

    def __next__(self):
        # print('index')
        # print(self.index_in_list)
        # stop when all indeces were given
        if self.global_index >= self.len:
            self._start()
            raise StopIteration

        # if we reached the end of the current list, we setup a new one.
        # The offset is needed to add the value of the previously seen indeces
        if self.index_in_list >= len(self.current_list):
            self.offset += len(self.current_list)

            # print('offset')
            # print(self.offset)
            # print('---')

            # get the next list
            self.list_index += 1
            self._setup_new_list()
            # start from the beginning in the new list
            self.index_in_list = 0

        output = self.current_list[self.index_in_list] + self.offset
        self.index_in_list += 1
        self.global_index += 1

        return output
