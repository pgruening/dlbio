import collections
import glob
import os
import warnings

import numpy as np

from . import helpers
from .helpers import file_path_to_ID, load_file_with_pre_processing, list_of_images_to_array

import random
import json


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

        for counter, (image_path, image_id) in enumerate(images):
            image = self.load_file(image_path, self.image_pre_processing, None)

            tmp_labels = [x[0] for x in labels if x[1] == image_id]
            if len(tmp_labels) == 0:
                warnings.warn(
                    'no labels found for {}. Skipping...'.format(image_id))
                continue

            if counter % 1000 == 0:
                print('{}/{} images processed'.format(counter, len(images)))

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

        # write statistics
        output_path = os.path.splitext(file_path)[0]
        out = {'averaged_mean': averaged_mean/counter,
               'averaged_stdev': averaged_stdev/counter}

        json_file_path = output_path+'_statistics.json'
        with open(json_file_path, 'w') as f:
            json.dump(out, f)
        print('Wrote json-file: {}.'.format(
            json_file_path))


class MultiDataFileCreator(DataFileCreator):
    def create_dataset_file(self, output_path, base_file_id, num_images=64):
        images = get_path_and_ids(self.image_folders, self.image_globs)
        labels = get_path_and_ids(self.label_folders, self.label_globs)

        random.shuffle(images)
        random.shuffle(labels)

        averaged_mean = 0.0
        averaged_stdev = 0.0
        counter = 0.0

        index = 0
        id_dict = {'base_path': output_path}
        while images:
            data = Data([])
            num_used_images = 0
            ids = []

            # collect images for file
            while num_used_images < num_images:

                if not images:
                    break

                (image_path, image_id) = images.pop()

                tmp_labels = _get_labels(labels, image_id)
                if tmp_labels is None:
                    continue

                image = self.load_file(
                    image_path, self.image_pre_processing, None)

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

                ids.append(image_id)
                num_used_images += 1

            # write file
            file_name = base_file_id + '_' + str(index).zfill(2) + '.npy'
            file_path = os.path.join(output_path, file_name)
            data.to_numpy_file(file_path)
            print('Wrote file: {} with {} images.'.format(
                file_path, num_used_images))

            id_dict[file_name] = ids
            index += 1

        # write ids to json file
        json_file_path = os.path.join(output_path, base_file_id+'.json')
        with open(json_file_path, 'w') as f:
            json.dump(id_dict, f)
        print('Wrote json-file: {}.'.format(
            json_file_path))

        # write statistics
        out = {'averaged_mean': averaged_mean/counter,
               'averaged_stdev': averaged_stdev/counter}

        json_file_path = os.path.join(
            output_path, base_file_id+'_statistics.json')
        with open(json_file_path, 'w') as f:
            json.dump(out, f)
        print('Wrote json-file: {}.'.format(
            json_file_path))


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
    folders : list of strings
        paths determining where to search for data
    match_strings : list of strings
        strings used for linux globbing (e.g. *.png for all png files)
    Returns
    -------
    tuples: (file_path, file_id)
    """
    output = []
    print(folders, match_strings)
    if len(folders) != len(match_strings):
        raise ValueError('Num folders does not equal num match strings')

    for (folder, match_string) in zip(folders, match_strings):
        glob_string = os.path.join(folder, match_string)
        files = glob.glob(glob_string)

        if len(files) == 0:
            warnings.warn(
                'No Ids for generated search: {}'.format(glob_string)
            )

        output.extend([(x, file_path_to_ID(x))
                       for x in files])
    if len(output) == 0:
        raise ValueError(
            'ERROR: Could not find any files in the given locations.' +
            ' Current working dir: {}'.format(os.getcwd())
        )
    return output


def _get_labels(labels, image_id):
    tmp_labels = [x[0] for x in labels if x[1] == image_id]
    if not tmp_labels:
        warnings.warn(
            'no labels found for {}. Skipping...'.format(image_id))
        return None
    return tmp_labels
