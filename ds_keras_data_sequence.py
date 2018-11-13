""" Code inspired by:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    https://keras.io/utils/#sequence
    The idea is to give the path to a pre-processed .npy file. Which will
    be held in RAM and can be accesed via the object. Since cross-validation
    will be used, the generate function will get a list of indeces indicating
    wich images can be selected for data_generation. Most importantly, all
    images need to be randomly cropped to a predifined size, but other
    augmentation techniques should also be possible.
"""
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import Sequence as keras_sequence

from .ds_Igenerator import IGenerator
from .global_constants import DATA_ID_INDEX, DATA_IMAGE_INDEX, DATA_LABEL_INDEX

# NOTE: DataSequence was once intended to be capable of loading several labels
# for one image and stacking them together. For now this has not been used and
# is therefore no working feature at the moment.
DEFAULT_DEBUG = False


class DataSequence(keras_sequence, IGenerator):
    def __init__(self, data,
                 augmentation_functions,
                 batch_size,
                 ** kwargs
                 ):
        """DataSequence that can returns items of data in batches.
        Can either return arrays (X, y, ID) or (X, y), where y can also be None.
        Size of X ans y is (b, h, w, num_dims) while ID is a list of length b
        Parameters
        ----------
        data : numpy array
            Array of arrays containing images, labels and ID-strings
        augmentation_functions : list of functions
            List of functions fcn(image, label) that return an augmented
            image and augmented label.

        kwargs:

        used_sample_id_list: list of str
            Pass a list of specific ids that can be found in data. Only those
            IDs will be used. A simple way to implement training, val and test
            set for one dataset
        num_id_repeats: int
            If the dataset is small each image can be repeated, therefore
            will be drawn n-times in an epoch
        return_ID: boolean
            If true the generator will return (X, y, ID) else only (X, y).
            Note that keras training only works with the latter.
        shuffle: boolean
            If true, each epoch the list of samples is randomly shuffled.
        batch_size:
            Define number of samples returned per iteration. The value b above.
        """
        # positional arguments
        self.data = data
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

        self._make_index_list(used_sample_id_list, num_id_repeats)
        self.id_list = copy.copy(used_sample_id_list)

        self.shuffle = shuffle
        self.batch_size = batch_size

        self.has_labels = len(self.data[0]) >= 3

        self.return_ID = return_ID

        if self.shuffle is True:
            np.random.shuffle(self.index_list)

    def _make_index_list(self, used_sample_id_list, num_id_repeats=1):
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
            self.index_list = [i for i in range(self.data.shape[0])
                               if self.data[i][DATA_ID_INDEX] in used_sample_id_list
                               ]

            if len(self.index_list) != len(used_sample_id_list):
                warnings.warn("Not all images found. \
                Found: {}, requested: {}".format(len(self.index_list),
                                                 len(used_sample_id_list))
                              )

        # for small datasets,
        #  the ids can be repeated to get a reasonable batch size working
        self.index_list = self.index_list*num_id_repeats

    def __getitem__(self, index):
        """Generates data of batch_size samples. x[i] equals x.__getitem(i)__"""

        # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        tmp_indeces = self.index_list[
            index*self.batch_size: (index+1)*self.batch_size]

        X = []
        IDs = []
        if self.has_labels:
            y = []
        else:
            y = None

        # Generate data
        for _i, load_index in enumerate(tmp_indeces):
            image = np.copy(self.data[load_index][DATA_IMAGE_INDEX])
            IDs.append(self.data[load_index][DATA_ID_INDEX])
            if self.DEBUG:
                print('load index: {}'.format(load_index))
                print('loaded image_shape: {}'.format(image.shape))

            # generator can also be used with test set
            if self.has_labels:
                label = np.copy(self.data[load_index][DATA_LABEL_INDEX:])
            else:
                label = None
            if self.DEBUG:
                print('label:')
                print(label.shape)

            # perform augmentation
            for func in self.augmentation_functions:
                if self.DEBUG:
                    print('func: {}'.format(func))
                try:
                    image, label = func(image, label)
                except ValueError as Error:
                    print(Error)
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

            X.append(image)
            if label is not None:
                if label.ndim == 2:
                    label = label[:, :, np.newaxis]
                y.append(label)

        X = self._to_stack(X)
        if y is not None:
            y = self._to_stack(y)

        if self.batch_size == 1:
            IDs = IDs[0]

        if self.DEBUG:
            print('X: {}'.format(X))

        if self.return_ID:
            return X, y, IDs
        else:
            return X, y

    def _to_stack(self, values):
        """Make sure values is a 4D vector of shape
        (b, h, w, dim)

        Parameters
        ----------
        values : list of arrays or array
            list of images (or labels) or a single image (or label).
        Returns
        -------
        np.array of shape (b, h, w, dim)
            stack input to 4D np.array

        """
        if self.batch_size > 1:
            try:
                values = np.stack(values, axis=0)
            except Exception as identifier:
                for x in values:
                    print(x.shape)
                    _, ax = plt.subplots(1)
                    ax.imshow(x[..., 0])
                    ax.set_title('ERROR!')
                    plt.show()
                print(identifier)
                raise(Exception)
        else:
            values = values[0][np.newaxis, ...]
        return values

    def print_ids(self):
        """Print out all ids that can be returned by the generator.

        """
        ids = [self.data[x][DATA_ID_INDEX] for x in self.index_list]
        print(ids)

    def get_id(self, ID):
        """Return sample of a specific ID

        Parameters
        ----------
        ID : str
            which sample to be returned
        Returns
        -------
        tuple of (np.array, np.array, str)
            Return the image, label and ID of the input ID.
        """
        for i in range(self.data.shape[0]):
            if self.data[i][DATA_ID_INDEX] == ID:
                return self.data[i]

    def set_used_samples(self, used_sample_id_list):
        """Change the generators sample list.

        Parameters
        ----------
        used_sample_id_list : list of str
            ID Strings given here are returned by the generator
        """
        self._make_index_list(used_sample_id_list)

    def __len__(self):
        """Return number of batches returned per one epoch.
        Can be called like this: len(data_sequence_object).

        Returns
        -------
        int
            returns num_indeces/batch_size
        """
        if self.batch_size == 1:
            return len(self.index_list)
        else:
            return max(1, len(self.index_list)//self.batch_size)

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.index_list)
        super(DataSequence, self).on_epoch_end()

    def print_augmentation_functions(self):
        """Print all augmentation functions that are currently used.

        """
        for func in self.augmentation_functions:
            print(func.__name__)

    def save_output(self,
                    image_save_fcn,
                    label_save_fcn=None
                    ):
        """Save the the images of each batch to an image and
        a label folder (if labels are defined).

        Parameters
        ----------
        image_save_fcn : function
            function takes in the image id and the image.
            The function must know where to save the image and 
            which file ending is used
        label_save_fcn : function, optional
            function takes in the image id and the image.
            The function must know where to save the image and 
            which file ending is used
            (the default is None, which means no labels are saved)

        """
        ctr = 0
        for i in range(len(self)):
            try:
                X, y, _ = self[i]
            except ValueError:
                X, y = self[i]

            for j in range(X.shape[0]):
                im_name = str(ctr).zfill(5)
                image_save_fcn(im_name, X[j, ...])

                if y is not None and label_save_fcn is not None:
                    label_save_fcn(im_name, y[j, ...])

                ctr += 1
