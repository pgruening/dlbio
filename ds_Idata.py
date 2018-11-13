import collections
import glob
import os
import warnings

import numpy as np

from . import helpers
from .helpers import file_path_to_ID, load_file_with_pre_processing, list_of_images_to_array


class DataFileCreator(object):
    def __init__(
        self,
        image_folders,
        image_globs,
        image_pre_processing,
        label_folders,
        label_globs,
        label_pre_processing=None,
        load_file=load_file_with_pre_processing
    ):
        self.image_folders = image_folders
        self.image_globs = image_globs
        self.image_pre_processing = image_pre_processing

        self.label_folders = label_folders
        self.label_globs = label_globs
        self.label_pre_processing = label_pre_processing

        self.load_file = load_file

    # TODO: check if file exists + overwrite FLAG
    def create_dataset_file(self, file_path):

        data = Data([])

        images = get_path_and_ids(self.image_folders, self.image_globs)
        labels = get_path_and_ids(self.label_folders, self.label_globs)

        averaged_mean = 0.0
        averaged_stdev = 0.0
        counter = 0.0
        output = []

        for (image_path, image_id) in images:
            image = self.load_file(image_path, self.image_pre_processing, None)

            tmp_labels = [x[0] for x in labels if x[1] == image_id]
            if len(tmp_labels) == 0:
                warnings.warn(
                    'no labels found for {}. Skipping...'.format(image_id))
                continue

            label_output = []
            for tmp_label in tmp_labels:
                label = self.load_file(
                    tmp_label, None, self.label_pre_processing)
                label_output.append(label)

            averaged_mean += np.mean(np.copy(image).astype('float32'))
            averaged_stdev += np.std(np.copy(image).astype('float32'))
            counter += 1.0

            data.dataset_tuples.append(
                DatasetTuple(image_id, image, label_output)
            )

        data.to_numpy_file(file_path)


DatasetTuple = collections.namedtuple('DatasetTuple', 'ID image label')


class Data(object):
    def __init__(self, dataset_tuples):
        self.dataset_tuples = dataset_tuples

    def to_numpy_file(self, file_path):
        output = [[x.ID, x.image] + x.label
                  for x in self.dataset_tuples]
        output = list_of_images_to_array(output)
        np.save(file_path, output)

    def from_numpy_file(self, file_path):
        pass


def get_path_and_ids(folders, match_strings):
    """Look for files matching the corresponding match string in a list
    of folders

    Parameters
    ----------
    folders : [type]
        [description]
    match_strings : [type]
        [description]
    Returns
    -------
    tuples: (file_path, file_id)
    """
    output = []
    print(folders, match_strings)
    for (folder, match_string) in zip(folders, match_strings):
        glob_string = os.path.join(folder, match_string)
        files = glob.glob(glob_string)

        if len(files) == 0:
            warnings.warn(
                'No Ids for generated search: {}'.format(glob_string)
            )

        output.extend([(x, file_path_to_ID(x))
                       for x in files])
    return output
