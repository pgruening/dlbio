import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class SegmentationDataset(Dataset):

    def __init__(self, *args, **kwargs):
        self.images = list()
        self.labels = list()
        self.to_tensor = ToTensor()
        self.image_aug = kwargs.get('im_aug', None)
        self.normalize = kwargs.get('normalize', None)

        self.len = kwargs.get('ds_len', -1)
        self.num_classes = kwargs.get('num_classes', -1)
        self.data_aug = kwargs.get(
            'data_aug',
            lambda x: print('implement data_aug!')
        )

        self.__ran_data_check = False

    def _run_data_check(self):
        expected_unique = np.arange(0, self.num_classes)
        expected_ndims = 2
        for y in self.labels:
            y_unique = np.unique(y.flatten())
            assert (y_unique == expected_unique).min(), print(
                f'{y_unique}!={expected_unique}')

            assert y.ndim == expected_ndims, f'expect shape (h, w), got {y.shape}'

        # all input images should have the same dimension
        x_dims = set()
        for x in self.images:
            if x.ndim == 2:
                x_dims.add(1)
            else:
                x_dims.add(x.shape[-1])

        assert len(x_dims) == 1

        self.__ran_data_check = True

    def __getitem__(self, index):
        assert self.__ran_data_check, 'Run self._run_data_check in your init!'
        index = self._compute_index(index)
        x, y = np.copy(self.images[index]), np.copy(self.labels[index])

        seed = np.random.randint(low=0, high=10e6)
        old_state = random.getstate()

        random.seed(seed)
        np.random.seed(seed)
        x = self.data_aug(x)
        if self.image_aug is not None:
            x = self.image_aug(x)

        x = self.to_tensor(x)

        if self.normalize is not None:
            x = self.normalize(x)

        # y = (h,w)
        random.seed(seed)
        np.random.seed(seed)

        y = self.data_aug(y)

        y = torch.tensor(np.array(y))
        y = y.long()
        random.setstate(old_state)

        return {'x': x, 'y': y}

    def _compute_index(self, index):
        index = index % len(self.images)
        return index

    def __len__(self):
        return self.len
