import copy
import json
import os
import random

from .ds_Idata import get_path_and_ids
import warnings


class CrossValidationGenerator(object):

    def __init__(self, cv_run_file_name, num_cvs=3, **kwargs):
        self.cv_run_file_name = cv_run_file_name
        self.current_run = 0
        self.cross_validation_runs = []
        self.num_cvs = num_cvs
        self.percent_val = kwargs.get('percent_val', .2)
        self.percent_test = kwargs.get('percent_test', .1)

    def setup_cross_validation(self):
        ids = self.get_sample_ids()

        num_test = int(float(len(ids))*self.percent_test)
        if num_test == 0:
            warnings.warn('With a test percentage of {}, no test sample is drawn'.format(
                self.percent_test))

        num_val = int(float(len(ids))*self.percent_val)
        if num_val == 0:
            warnings.warn('With a validation percentage of {}, no validation sample is drawn'.format(
                self.percent_val))

        def take_n(seq, n):
            output = []
            for _i in range(n):
                element = random.choice(seq)
                output.append(element)
                seq.remove(element)
            return output

        for _n in range(self.num_cvs):
            tmp_ids = copy.copy(ids)

            test = take_n(tmp_ids, num_test)
            val = take_n(tmp_ids, num_val)

            cv_dict = {'train': tmp_ids, 'val': val, 'test': test}
            self.cross_validation_runs.append(cv_dict)

    def get_sample_ids(self):
        """If you want to use standard setup_cross_validation,
        this needs to be implemented. If you overload setup_cross_validation,
        you might not need this function at all.

        """
        raise NotImplementedError('You use setup_cross_validation. Therefore,'
                                  + ' this function needs to be implemented.')

    def save_runs(self, path_to_experiment_folder):
        path_to_file = os.path.join(
            path_to_experiment_folder, self.cv_run_file_name)
        with open(path_to_file, 'w') as f:
            json.dump(self.cross_validation_runs, f)

    def load_runs(self, folder):
        path_to_file = os.path.join(
            folder, self.cv_run_file_name)
        with open(path_to_file, 'r') as f:
            self.cross_validation_runs = json.load(f)

    def get_next_run(self):
        self.current_run += 1
        if (self.current_run - 1) < len(self.cross_validation_runs):
            return self.cross_validation_runs[self.current_run-1]
        else:
            raise ValueError("No next run available.")

    def get_run(self, index):
        return self.cross_validation_runs[index]

    def get_num_cvs(self):
        return self.num_cvs

    def reset_counter(self):
        self.current_run = 0


def get_sample_ids(image_folders,
                   image_match_strings,
                   label_folders,
                   label_match_strings):
    """Return all image ids that can be found in the image and
    label folders.

    Parameters
    ----------
    image_folders : list of str
        where to look for images
    image_match_strings : list of str
        glob strings for each image folder
    label_folders : list of str
        where to look for corresponding labels
    label_match_strings : list of str
        glob strings for each label folder
    Returns
    -------
    list of strings
        see above
    """

    images = get_path_and_ids(
        image_folders, image_match_strings)
    image_ids = set([x[1] for x in images])

    labels = get_path_and_ids(
        label_folders, label_match_strings)
    label_ids = set([x[1] for x in labels])
    return list(image_ids.intersection(label_ids))
