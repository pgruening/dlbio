import copy
import warnings

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .ds_Igenerator import IGenerator
from .global_constants import DATA_ID_INDEX, DATA_IMAGE_INDEX, DATA_LABEL_INDEX
import torchvision.transforms as transforms


class DataSequence(IGenerator):
    def __init__(self,
                 data,
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

        self.dataset = PyTorchDataset(data,
                                      augmentation_functions,
                                      used_sample_id_list,
                                      num_id_repeats,
                                      return_ID,
                                      DEBUG,
                                      to_pytorch_tensor)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers

        )

    # def __getitem__(self, index):
    #    return self.data_loader.__getitem__(index)

    def __iter__(self):
        return self.data_loader.__iter__()

    def __len__(self):
        return self.data_loader.__len__()


class PyTorchDataset(Dataset):
    def __init__(self,
                 data,
                 augmentation_functions,
                 used_sample_id_list,
                 num_id_repeats,
                 return_ID,
                 DEBUG,
                 to_pytorch_tensor
                 ):

        self.data = data
        self.augmentation_functions = augmentation_functions

        self.has_labels = len(self.data[0]) >= 3
        self.return_ID = return_ID

        self._make_index_list(used_sample_id_list, num_id_repeats)
        self.id_list = copy.copy(used_sample_id_list)

        self.DEBUG = DEBUG

        self.to_pytorch_tensor = to_pytorch_tensor
        self.to_tensor = transforms.ToTensor()

    def _make_index_list(self, used_sample_id_list, num_id_repeats):
        """ Is used if the sequence should only work on a subset of the data.
        Computes an list of integer indeces from a given list
        of ID strings.

        Parameters
        ----------
        used_sample_id_list : list of str
            list containing ids as can be found in self.data[i][DATA_ID_INDEX]
        num_id_repeats : int, optional
            the list is concatenated num_id_repeats times. (the default is 1,
             which means the list is not changed.)

        """
        if used_sample_id_list is None:
            self.index_list = [i for i in range(self.data.shape[0])]

        else:
            def is_match(
                i): return self.data[i][DATA_ID_INDEX] in used_sample_id_list

            self.index_list = [i for i in range(
                self.data.shape[0]) if is_match(i)]

            if len(self.index_list) != len(used_sample_id_list):
                warnings.warn("Not all images found. \
                Found: {}, requested: {}".format(len(self.index_list),
                                                 len(used_sample_id_list))
                              )

        # for small datasets,
        #  the ids can be repeated to get a reasonable batch size working
        self.index_list = self.index_list*num_id_repeats

    def __getitem__(self, index):
        index = self.index_list[index]

        image = np.copy(self.data[index][DATA_IMAGE_INDEX])
        image_id = self.data[index][DATA_ID_INDEX]
        if self.has_labels:
            label = np.copy(self.data[index][DATA_LABEL_INDEX:])

        # perform augmentation
        for func in self.augmentation_functions:
            if self.DEBUG:
                print('func: {}'.format(func))
            try:
                image, label = func(image, label)
            except ValueError as error_:
                print(error_)
                print('-'*15)
                raise ValueError(
                    'Error in function {} for input {} and {}'.format(
                        func, image, label
                    )
                )
            if self.DEBUG:
                print('image_shape after aug: {}'.format(image.shape))
                print('label_shape after aug: {}'.format(label.shape))
                print(np.max(label))

        if image.ndim == 2:
            image = image[..., np.newaxis]

        if label is not None and label.ndim == 2:
            label = label[:, :, np.newaxis]

        # NOTE: pytorche uses (b,c,h,w) all functions are in tf's (b,h,w,c)
        if self.to_pytorch_tensor:
            image = self.to_tensor(image)
            label = self.to_tensor(label)

        if self.return_ID:
            return image, label, image_id
        else:
            return image, label

    def __len__(self):
        return len(self.index_list)

# TODO add dataloader
