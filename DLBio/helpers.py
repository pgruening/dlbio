import collections
import csv
import datetime
import glob
import inspect
import json
import os
import random
import re
import shutil
import time
import warnings
from collections import namedtuple
from datetime import datetime
from os.path import isfile, join, splitext

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from PIL import Image

SRC_COPY_FOLDER = ''
TB_LOG_FOLDER = ''


def check_mkdir(directory_or_file, is_dir=False):
    if is_dir:
        # NOTE: useful if the directory has a point in the foldername
        directory = directory_or_file
    else:
        directory = _get_directory(directory_or_file)

    if not os.path.isdir(directory):
        os.makedirs(directory)


def save_options(file_path, options):
    warnings.warn(
        'helpers.save_options is deprecated and has been moved to pytorch_helpers'
    )
    if not hasattr(options, "__dict__"):
        out_dict = dict(options._asdict())
    else:
        out_dict = options.__dict__

    # add the current time to the output
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

    out_dict.update({
        'start_time': dt_string
    })

    with open(file_path, 'w') as file:
        json.dump(out_dict, file)


def load_json(file_path):
    if not isfile(file_path):
        return None

    with open(file_path, 'r') as file:
        out = json.load(file)
    return out


def get_sub_dataframe(df, cols_and_vals, bool_fcn=None, return_where=False):
    """Select a sub-dataframe according to cols_and_val, and maybe a specified
    boolean function. Each key of cols_and_val denotes one column that is used
    for the selection process. Only rows are kept where the dataframe values at
    key are equal to the values in cols_and_val[key].

    Parameters
    ----------
    df : pd.DataFrame
    cols_and_vals : dict {key:obj} or {key:[obj1, obj2, ...]}
        keys correspond to the respective df columns. By default, a df-row is 
        kept if its value at key is equal to obj or in the list 
        [obj1, obj2, ...]
    bool_fcn : function that returns bool, optional
        function that returns a boolean based on the dataframe column as first
        input and the value as second input, and key as third input. 
    return_where: bool
        return the boolean list instead of the the reduced dataframe
    """
    def select(df_column, value, key):
        if bool_fcn is None:
            return df_column == value
        else:
            return bool_fcn(df_column, value, key)

    df = df.copy()
    where_and = np.ones(df.shape[0]) > 0
    for key, values in cols_and_vals.items():
        # select all rows i where df[key][i] in values
        if isinstance(values, list):
            _where_or = np.ones(df.shape[0]) == 0
            for v in values:
                tmp = np.array(select(df[key], v, key))
                _where_or = np.logical_or(_where_or, tmp)
        else:
            v = values
            _where_or = np.array(select(df[key], v, key))

        where_and = np.logical_and(where_and, _where_or)

    if return_where:
        return where_and
    else:
        return df[where_and]


def copy_source(out_folder, max_num_files=100, do_not_copy_folders=None):
    """Copies the source files of the current working dir to out_folder

    Parameters
    ----------
    out_folder : str
        where to copy the files
    max_num_files : int, optional
        makes sure not to copy files in endless loop, by default 100
    do_not_copy_folders : list of str, optional
        list of folder names that are not supposed to be copied, by default None
    """
    if do_not_copy_folders is None:
        warnings.warn('No folders excluded from source_copy! Are you sure?')
        do_not_copy_folders = []

    do_not_copy_folders += ['__pycache__']

    out_f = out_folder.split('/')[-1]
    out_folder = join(out_folder, 'src_copy')
    print(f'Source copy to folder: {out_folder}')
    ctr = 0
    for root, _, files_ in os.walk('.'):

        # remove the base folder:
        # e.g. './models' or '.' -> grab anything behind that
        current_folder = re.match(r'^(.$|.\/)(.*)', root).group(2)

        # NOTE: only looks at the root folder, maybe adjustments are needed
        # at some point
        do_continue = False
        for x in current_folder.split('/'):
            if x in do_not_copy_folders:
                do_continue = True
                break
        if do_continue:
            continue

        # do not copy anything that is on the path to the output folder (out_f)
        if out_f in root.split('/'):
            continue

        # make sure there is no never-ending copy loop!!
        tmp = [1 for x in current_folder.split('/') if x in ['src_copy']]
        if tmp:
            continue

        files_ = [x for x in files_ if splitext(x)[-1] in ['.py']]
        if not files_:
            continue
        else:
            print(f'src_copy from folder: {root}')

        for file in files_:
            dst = file
            if current_folder != '':
                dst = join(current_folder, file)

            dst = join(out_folder, dst)
            src = join(root, file)

            check_mkdir(dst)
            shutil.copy(src, dst)

            ctr += 1
            assert ctr < max_num_files, 'too many files copied.'


def search_in_all_subfolders(rgx, folder, search_which='files', match_on_full_path=False, depth=None):
    # TODO: rename to find
    def is_rgx_match(rgx, x):
        return bool(re.fullmatch(rgx, x))
    outputs = []
    assert os.path.isdir(folder), f'folder not found: {folder}'

    for root, dirs_, files_ in os.walk(folder):
        if depth is not None:
            # Don't search to deep into the folder tree
            # remove base path
            tmp = root[len(folder):]
            # count folders
            current_depth = len(tmp.split('/')) - 1
            if current_depth > depth:
                continue

        if search_which == 'files':
            to_search = files_
        elif search_which == 'dirs':
            to_search = dirs_
        elif search_which == 'all':
            to_search = files_ + dirs_
        else:
            raise ValueError(f'unknown search_type: {search_which}')

        if not to_search:
            continue

        if match_on_full_path:
            tmp = [
                join(root, x) for x in to_search if is_rgx_match(rgx, join(root, x))
            ]
        else:
            tmp = [join(root, x) for x in to_search if is_rgx_match(rgx, x)]

        outputs += tmp

    return outputs


def search_rgx(rgx, path):
    return [x for x in os.listdir(path) if bool(re.match(rgx, x))]


def get_from_module(py_module, bool_fcn):
    """Check all objects of a python module and return all for which the
    bool function returns true

    Parameters
    ----------
    py_module : python module
        import x -> pass x
    bool_fcn : function that returns a boolean
        y = fcn(x) -> x = object found in py_module, y in [True, False]

    Returns
    -------
    list of tuples (name (str), object)
    """
    cls_members = inspect.getmembers(py_module, bool_fcn)
    return cls_members


def get_subfolders(base_folder):
    # TODO: rename to find
    return next(os.walk(base_folder))[1]


def get_parent_folder(folder):
    return '/'.join(folder.split('/')[:-1])


def dict_to_options(opt_dict):
    """Transforms a dictionary into an options object,
    similar to the object created by the ArgumentParser. 

    Parameters
    ----------
    opt_dict : dict

    Returns
    -------
    object
        "key: value" -> object.key == value
    """
    Options = namedtuple('Options', opt_dict.keys())

    return Options(**opt_dict)


class MyDataFrame():
    def __init__(self, verbose=0):
        self.x = dict()
        self.max_num_items = 0
        self.verbose = verbose

    def update(self, in_dict, add_missing_values=False, missing_val=np.nan):
        for k, v in in_dict.items():

            if isinstance(v, list):
                warnings.warn(f'Input for {k} is list, consider add_col.')

            if k not in list(self.x.keys()):
                if self.verbose > 0:
                    print(f'added {k}')
                # case 1: df just intialized
                if self.max_num_items == 0:
                    self.x[k] = [v]
                else:
                    # case 2: entire new key is added
                    if add_missing_values:
                        # fill with missing values to current num items
                        self.x[k] = [missing_val] * self.max_num_items
                        self.x[k].append(v)

            else:
                self.x[k].append(v)

        if add_missing_values:
            self._add_missing(missing_val)

    def _add_missing(self, missing_val):
        self._update()
        for k in self.x.keys():
            if self.verbose > 1 and len(self.x[k]) < self.max_num_items:
                print(f'add missing: {k}')

            while len(self.x[k]) < self.max_num_items:
                self.x[k].append(missing_val)

    def _update(self):
        self.max_num_items = max([len(v) for v in self.x.values()])

    def add_col(self, key, col):
        self.x[key] = col

    def get_df(self, cols=None):
        assert self._check_same_lenghts()
        return pd.DataFrame(self.x, columns=cols)

    def _check_same_lenghts(self):
        len_vals = {k: len(v) for k, v in self.x.items()}
        if len(set(len_vals.values())) > 1:
            print(len_vals)
            return False

        return True


def set_plt_font_size(font_size):
    # font = {'family' : 'normal',
    #    'weight' : 'bold',
    #    'size'   : 22}
    font = {'size': font_size}
    matplotlib.rc('font', **font)


class ToBin():
    def __init__(self, n):
        self.n = n

    def __call__(self, arr):
        if isinstance(arr, int):
            return np.stack(self._to_bin(arr), 0)
        assert arr.ndim == 1
        out = [self._to_bin(int(x)) for x in list(arr)]
        return np.stack(out, 0)

    def _to_bin(self, x):
        return np.array([float(s) for s in self._bin(x)])

    def _bin(self, x):
        return format(x, 'b').zfill(self.n)


def get_dataframe_from_row(df, index):
    row = dict(df.iloc[index])
    return pd.DataFrame({k: [v] for k, v in row.items()})


def get_parent_folder(directory_or_file):
    directory = _get_directory(directory_or_file)
    return directory.split('/')[-1]


def _get_directory(directory_or_file):
    if os.path.splitext(directory_or_file)[-1] != '':
        directory = '/'.join(directory_or_file.split('/')[:-1])
    else:
        directory = directory_or_file
    return directory


def find_image(im_path, labels_):
    found_ = [x for x in labels_ if is_match(im_path, x)]
    if not found_:
        return None
    assert len(found_) == 1
    return found_[0]


def get_id(filepath_or_filename):
    # NOTE: may not works with files like name.sth.png
    x = os.path.basename(filepath_or_filename)
    return os.path.splitext(x)[0]


def is_match(file_path_x, file_path_y):
    return get_id(file_path_x) == get_id(file_path_y)


def open_npy():
    files_ = glob.glob('*.npy')
    file = random.choice(files_)
    file = np.load(file)
    print(file.shape)
    plt.imshow(file)
    plt.show()


def read_flow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename, uv, v=None):
    tag_character = np.array([202021.25], np.float32)

    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(tag_character)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


class IRectangle(object):
    x_pos = 0
    y_pos = 1
    w_pos = 2
    h_pos = 3

    def __init__(self, **kwargs):
        self.x = kwargs['x']
        self.y = kwargs['y']
        self.h = kwargs['h']
        self.w = kwargs['w']

    def get_pyplot_patch(self):
        raise NotImplementedError


class Rectangle(IRectangle):

    def estimate_jaccard_index(self, rectangle):
        # estimate position of intersection rectangle
        left = max(self.x, rectangle.x)
        top = max(self.y, rectangle.y)
        right = min(self.x + self.w, rectangle.x + rectangle.w)
        bottom = min(self.y + self.h, rectangle.y + rectangle.h)

        a = max(right - left, 0)
        b = max(bottom - top, 0)

        # estimate areas
        area_intersection = a * b
        area_union = self.w * self.h + rectangle.w * rectangle.h - area_intersection

        jaccard_index = safe_division(area_intersection, area_union)
        return jaccard_index

    def get_array(self):
        output = np.zeros(4)
        output[self.x_pos] = self.x
        output[self.y_pos] = self.y
        output[self.w_pos] = self.w
        output[self.h_pos] = self.h
        return output

    def to_c_rectangle(self):
        cx = self.x + .5 * self.w
        cy = self.y + .5 * self.h
        return cRectangle(
            cx=cx,
            cy=cy,
            w=self.w,
            h=self.h,
        )

    def get_pyplot_patch(self, image_shape, jitter=0., color='r'):
        x = self.x + jitter
        y = self.y + jitter
        h = self.h + jitter
        w = self.w + jitter

        xy = [x * image_shape[1], y * image_shape[0]]
        h_box = h * image_shape[1]
        w_box = w * image_shape[0]
        return patches.Rectangle(
            xy, w_box, h_box,
            linewidth=9,
            edgecolor=color,
            facecolor='none')

    def to_p_rectangle(self, h, w):
        x = int(np.round(self.x * w))
        y = int(np.round(self.y * h))
        h_new = int(np.round(self.h * h))
        w_new = int(np.round(self.w * w))

        return pRectangle(x=x, y=y, h=h_new, w=w_new)


class cRectangle(IRectangle):
    def __init__(self, **kwargs):
        self.cx = kwargs['cx']
        self.cy = kwargs['cy']
        self.h = kwargs['h']
        self.w = kwargs['w']

    def to_rectangle(self, *args):
        x = self.cx - .5 * self.w
        y = self.cy - .5 * self.h
        return Rectangle(
            x=x,
            y=y,
            w=self.w,
            h=self.h,
        )

    def to_p_rectangle(self, h, w):
        rect = self.to_rectangle()
        return rect.to_p_rectangle(h, w)

    def estimate_jaccard_index(self, c_rectangle):
        rect1 = self.to_rectangle()
        rect2 = c_rectangle.to_rectangle()
        return rect1.estimate_jaccard_index(rect2)

    def get_pyplot_patch(self, image_shape, jitter=0., color='r'):
        rect = self.to_rectangle()
        return rect.get_pyplot_patch(image_shape, jitter, color)


class ConfidenceCRectangle(cRectangle):
    def __init__(self, **kwargs):
        self.confidence = kwargs.pop('confidence', 1.0)
        super(ConfidenceCRectangle, self).__init__(**kwargs)


class pRectangle(IRectangle):

    def estimate_jaccard_index(self, p_rectangle):
        rect1 = Rectangle(**self.__dict__)
        rect2 = Rectangle(**p_rectangle.__dict__)
        return rect1.estimate_jaccard_index(rect2)

    def get_pyplot_patch(self, jitter=0.0, color='r'):
        x = self.x + jitter
        y = self.y + jitter
        h = self.h + jitter
        w = self.w + jitter

        return patches.Rectangle(
            [x, y], w, h,
            linewidth=1,
            edgecolor=color,
            facecolor='none')

################### miscellaneous functions #############################################################


def save_image(file_path, array):
    image = Image.fromarray(array)
    image.save(file_path)


def load_image(image_path, pre_processing_fcn):
    """Simple image loader for cell images.
    If there is a pre_processing function, don't forget
    your pre_processing function.

    Parameters
    ----------
    image_path : str
      path to image file.
    pre_processing_fcn : function
      Function that is applied to the image after loading.
      If the value is None the image is returned without
      pre-processing.
    Returns
    -------
    numpy array of shape (h, w, c)
      Loaded image. If no pre-processing is done
      the image is of type uint8.
    """
    image = Image.open(image_path)
    image = np.array(image)

    # image is loaded with transparency value, we remove it
    if image.shape[-1] == 4:
        image = image[..., :3]
    if pre_processing_fcn is not None:
        image = pre_processing_fcn(image)
    return image


def to_uint8_image(image, eps=0.):
    """Rescale and cast image to uint8 format
    so it can be written to a png or jpg file

    Parameters
    ----------
    image : numpy array
        Image to transform
    eps: float
        add > 0 to prevent divide by zero errors
    Returns
    -------
    numpy array of type uint8
      Image is rescaled to [0,255] and casted.
    """
    if image.dtype == 'uint8':
        return image

    if image.dtype != 'float32':
        image = image.astype('float32')

    image -= np.min(image)
    image /= (np.max(image) + eps)
    return (255 * image).astype('uint8')


def sigmoid(x):
    return (1.0 + np.exp(-x))**-1.0


def softmax(input):
    """Softmax function without numerical stability!
    No batch processing.
    Parameters
    ----------
    input : 4- or 3D Tensor (b,h,w,c) or (h,w,c)
      Normally a neural network output. b needs to be one.
    Raises
    ------
    ValueError
      If the batchsize is greater than one.

    Returns
    -------
    3D Tensor
      Softmax output of the tensor.
    """
    # from batch to logit image
    if input.ndim == 4:
        if input.shape[0] > 1:
            raise ValueError('Too many input images')
        else:
            input = input[0, :, :, :]

    for n in range(input.shape[2]):
        input[:, :, n] = np.exp(input[:, :, n])

    denominator = np.sum(input, axis=2)
    for n in range(input.shape[2]):
        input[:, :, n] /= denominator
    return input


def make_output_image(connected_components_image):
    """Colormapping to connected components image.

    Parameters
    ----------
    connected_components_image : 2D numpy array with integers (h,w)
      image with each pixel encoding the belonging to a cell
    Returns
    -------
    uint8 numpy array of shape (h, w, 3)
      image with specific color pattern that can be saved.
    """
    connected_components_image = cell_labeling(connected_components_image)

    h, w = connected_components_image.shape
    output_image = np.zeros((h, w, 3), dtype='uint8')

    max_val = np.max(connected_components_image)
    no_cell_in_image = max_val == 0
    if no_cell_in_image:
        return output_image

    step = 3 * 255 // np.max(connected_components_image)
    indeces = np.unique(connected_components_image)
    np.random.shuffle(indeces)
    for n, i in enumerate(indeces):
        if i == 0:
            continue
        value = n * step

        tmp_image = np.zeros((h, w), dtype='uint8')
        for j in range(3):
            if value > 255:
                pixel_value = 0
                if np.random.rand() < .5:
                    pixel_value = 255
                tmp_image[connected_components_image == i] = pixel_value
                output_image[:, :, j] += tmp_image
                value -= 255
            else:
                tmp_image[connected_components_image == i] = value
                output_image[:, :, j] += tmp_image
                break

    return output_image


def get_time_string():
    """Get current time as a string.
    E.g. used to make a file ID
    Returns
    -------
    str
      current time in month, day, hour, minute
    """
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(
        timestamp).strftime('%m_%d_%H_%M')
    return timestamp


def list_of_images_to_array(input_list):
    """
      Workaround for numpy issue
      see https://github.com/numpy/numpy/issues/7453
    """
    input_list.insert(0, np.array([]))
    return np.array(input_list)[1:]


def plot_evaluation_image(input, filename):
    """Save a network evalutation image.

    Parameters
    ----------
    input : list of images
      list with (input_image, network_output, ground_truth)
    filename : str
      where to save the file.
    """
    _, ax = plt.subplots(1, 3)

    for i in range(3):
        ax[i].imshow(input[i])
    plt.title(input[-1])
    plt.savefig(filename)
    plt.close()


def cell_labeling(connected_component_image, do_shuffe=False):
    """Relabel cc_image so that the indeces
    are (1,2,3,...,number_of_cells)

    Parameters
    ----------
    image : connected_component_image
      image with each pixel encoding the belonging to a cell
    Returns
    -------
    connected_component_image
      Image with relabelled indeces.
    """
    # changes cell labels to (1...ncells)
    unique_values = np.unique(connected_component_image)
    if 0 not in unique_values:
        connected_component_image[0, 0] = 0
        unique_values = np.unique(connected_component_image)

    if np.any(unique_values - np.floor(unique_values)) > 0:
        warnings.warn(
            'Input might not be a connected_components image.\
             Values are no integers.')

    if len(unique_values) == 1:
        warnings.warn('Empty image, no connected components')
        output = np.zeros(connected_component_image.shape,
                          dtype=connected_component_image.dtype)
        return output

    if do_shuffe:
        unique_values = list(unique_values)
        unique_values.remove(0)
        random.shuffle(unique_values)
        unique_values.append(0)
        unique_values.reverse()

    # annoying warning when used with cropped data...
    # if np.shape(unique_values)[0] < 5:
    #    warnings.warn(
    #        'Input might not be a connected_components image.\
    #         Less than five unique values in the image.')
    output = np.zeros(connected_component_image.shape,
                      dtype=connected_component_image.dtype)
    for new_index, current_index in enumerate(unique_values):
        output[connected_component_image == current_index] = new_index

    return output


##################### mean average precision metric #################################################################
AVERAGE_PRECISION_THRESHOLD_VALUES = [
    .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]


def compute_mean_average_precision(input_pred, input_gt):
    """Mean average precision of two images: a prediction and a target.
    See https://www.kaggle.com/c/data-science-bowl-2018#evaluation for more
    information.
    Parameters
    ----------
    input_pred : cc_image
      prediction of an instance segmentation
    input_gt : binary or cc_image
      ground truth image

    Returns
    -------
    float
      mean average precision of the two inputs.
    """

    cc_image_pred = cell_labeling(input_pred)
    num_labels_pred = int(np.max(input_pred))
    stats_pred = compute_connected_component_stats(input_pred)
    cc_image_gt = cell_labeling(input_gt)
    num_labels_gt = int(np.max(cc_image_gt))
    stats_gt = compute_connected_component_stats(input_gt)

    # compute jaccard indeces according to bounding boxes
    if num_labels_pred == 0 or num_labels_gt == 0:
        return 0.0

    JI_Matrix_bounding_box = np.zeros((num_labels_pred, num_labels_gt))

    # 0 is considered background
    for i in range(1, num_labels_pred + 1):

        rectangle_pred = Rectangle(
            x=stats_pred[i - 1, cv2.CC_STAT_LEFT],
            y=stats_pred[i - 1, cv2.CC_STAT_TOP],
            w=stats_pred[i - 1, cv2.CC_STAT_WIDTH],
            h=stats_pred[i - 1, cv2.CC_STAT_HEIGHT]
        )

        for j in range(1, num_labels_gt + 1):
            rectangle_gt = Rectangle(
                x=stats_gt[j - 1, cv2.CC_STAT_LEFT],
                y=stats_gt[j - 1, cv2.CC_STAT_TOP],
                w=stats_gt[j - 1, cv2.CC_STAT_WIDTH],
                h=stats_gt[j - 1, cv2.CC_STAT_HEIGHT]
            )

            JI_Matrix_bounding_box[i - 1, j -
                                   1] = rectangle_gt.estimate_jaccard_index(
                                       rectangle_pred
            )

    # *greedy matching, then computing the real JI*

    # keep track of already matched ground truth cells
    iou_values_for_each_gt = -np.ones(num_labels_gt)

    matching_done = (num_labels_gt) == 0
    while not matching_done:
        maximum_index = np.unravel_index(JI_Matrix_bounding_box.argmax(),
                                         JI_Matrix_bounding_box.shape
                                         )

        JI_value = JI_Matrix_bounding_box[maximum_index[0], maximum_index[1]]

        matching_done = JI_value == -1
        if matching_done:
            break

        # compute real jaccard index for the two components
        # row -> prediction components, column ground_truth components
        pred_component = np.copy(cc_image_pred) == (maximum_index[0] + 1)
        gt_component = np.copy(cc_image_gt) == (maximum_index[1] + 1)
        iou_values_for_each_gt[maximum_index[1]] = simple_IOU(
            pred_component, gt_component)

        # Debugging, does matching work?
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(pred_component)
        # ax[1].imshow(gt_component)
        # ax[0].set_title(iou_values_for_each_gt[maximum_index[1]])
        # plt.show()
        # plt.close()

        # set already matched components -1
        JI_Matrix_bounding_box[:, maximum_index[1]] = -1.0
        JI_Matrix_bounding_box[maximum_index[0], :] = -1.0

    # print(iou_values_for_each_gt)

    # if there are more pred_cells than gt_cells, there are FPs
    FP = max(0, num_labels_pred - num_labels_gt)

    # num_labels_gt is either TP or FN, number increases with FPs
    num_all_cells = FP + num_labels_gt

    precision = 0.0
    for thres in AVERAGE_PRECISION_THRESHOLD_VALUES:
        TP_t = np.sum(iou_values_for_each_gt > thres)
        precision += safe_division(TP_t, num_all_cells)

    mean_average_precision = safe_division(
        precision, len(AVERAGE_PRECISION_THRESHOLD_VALUES))

    return mean_average_precision


def compute_connected_component_stats(input):
    """Get stats of the cc_image, similar to
    open_cv's connected components with stats.
    https://www.programcreek.com/python/example/89340/cv2.connectedComponentsWithStats
    Parameters
    ----------
    image : connected_component_image
      image with each pixel encoding the belonging to a cell
    Returns
    -------
    list of statistics
      return list of list with [(leftmost_pixel, topmost_pixel, width, height), ...]
      basically the rectangle (y,x,w,h) containing all pixels of a component
      ! CC_STAT_WIDTH = 2
      ! CC_STAT_HEIGHT = 3
    """
    max_num_columns = 1 + \
        np.max(np.asarray([cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                           cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]))
    # Python 2 code:
    #unique_indeces = map(lambda x: int(x), list(np.unique(input)))
    unique_indeces = [int(x) for x in list(np.unique(input))]
    output = np.zeros((len(unique_indeces) - 1, max_num_columns))
    for j, index in enumerate(unique_indeces):
        # 0 is background
        if index == 0:
            continue
        cell_indeces = np.argwhere(input == index)
        output[j - 1, cv2.CC_STAT_LEFT] = np.min(cell_indeces[:, 1])
        output[j - 1, cv2.CC_STAT_TOP] = np.min(cell_indeces[:, 0])

        output[j - 1, cv2.CC_STAT_WIDTH] = np.max(
            cell_indeces[:, 1]) - output[j - 1, cv2.CC_STAT_LEFT] + 1

        output[j - 1, cv2.CC_STAT_HEIGHT] = np.max(
            cell_indeces[:, 0]) - output[j - 1, cv2.CC_STAT_TOP] + 1
    return output


def slow_compute_connected_component_stats(input):
    """Get stats of the cc_image, similar to
    open_cv's connected components with stats.
    https://www.programcreek.com/python/example/89340/cv2.connectedComponentsWithStats
    Parameters
    ----------
    image : connected_component_image
      image with each pixel encoding the belonging to a cell
    Returns
    -------
    list of statistics
      return list of list with [(leftmost_pixel, topmost_pixel, width, height), ...]
      basically the rectangle (y,x,w,h) containing all pixels of a component
      ! CC_STAT_WIDTH = 2
      ! CC_STAT_HEIGHT = 3
    """
    max_num_columns = 1 + \
        np.max(np.asarray([cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                           cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]))
    unique_indeces = map(lambda x: int(x), list(np.unique(input)))
    index_to_cell_map = {str(x): i - 1 for i, x in enumerate(unique_indeces)}
    output = -1.0 * np.ones((len(unique_indeces) - 1, max_num_columns))
    for y in range(input.shape[0]):
        for x in range(input.shape[1]):

            cell_index = input[y, x, 0]
            if cell_index == 0:
                continue
            j = index_to_cell_map[str(cell_index)]
            current_min_x = output[j, cv2.CC_STAT_LEFT]
            if x < current_min_x or current_min_x == -1:
                output[j, cv2.CC_STAT_LEFT] = x

            current_max_x = output[j, cv2.CC_STAT_WIDTH]
            if x > current_max_x:
                output[j, cv2.CC_STAT_WIDTH] = x

            current_min_y = output[j, cv2.CC_STAT_TOP]
            if y < current_min_y or current_min_y == -1:
                output[j, cv2.CC_STAT_TOP] = y

            current_max_y = output[j, cv2.CC_STAT_HEIGHT]
            if y > current_max_y:
                output[j, cv2.CC_STAT_HEIGHT] = y

    # width and height is just set temporalily to maximum values
    for j in range(output.shape[0]):
        output[j, cv2.CC_STAT_WIDTH] -= output[j, cv2.CC_STAT_LEFT]
        output[j, cv2.CC_STAT_HEIGHT] -= output[j, cv2.CC_STAT_TOP]

    return output


def estimate_jaccard_index(rectangle_1, rectangle_2):
    """Return the jaccard index(IOU) of two rectangles

    Parameters
    ----------
    rectangle_1 : list with values (x,y,h,w)
    rectangle_2 : list with values (x,y,h,w)
    Returns
    -------
    float
      IOU of the two inputs.
    """
    x_1, y_1, w_1, h_1 = rectangle_1
    x_2, y_2, w_2, h_2 = rectangle_2

    # estimate position of intersection rectangle
    left = max(x_1, x_2)
    top = max(y_1, y_2)
    right = min(x_1 + w_1, x_2 + w_2)
    bottom = min(y_1 + h_1, y_2 + h_2)

    a = max(right - left, 0)
    b = max(bottom - top, 0)

    # estimate areas
    area_intersection = a * b
    area_union = w_1 * h_1 + w_2 * h_2 - area_intersection

    jaccard_index = safe_division(area_intersection, area_union)
    return jaccard_index


def simple_IOU(binary_image_1, binary_image_2):
    """IOU of two binary_images

    Parameters
    ----------
    binary_image_1 : numpy array of type float or int or bool 
      values are either one or zero. With one decoding,
      'belongs to object'
    binary_image_2 : numpy array of type float or int or bool 
      values are either one or zero. With one decoding,
      'belongs to object'
    Returns
    -------
    float
      IOU of the two images.
    """
    try:
        intersection = np.sum(binary_image_1 * binary_image_2)
        union = np.sum((binary_image_1 + binary_image_2).clip(max=1))
    except:
        plt.imshow(binary_image_1)
        plt.imshow(binary_image_2)
        plt.show()
    return safe_division(intersection, union)


################ Folder operations #################################


def make_src_copy(path_to_experiment_folder):
    path_to_copy_folder = os.path.join(
        path_to_experiment_folder, SRC_COPY_FOLDER)
    src_to_copy = []
    dst_to_copy = []
    for root, _, files in os.walk('./src'):
        if files is None:
            continue
        prefix = "".join(root.split('./src'))
        if len(prefix) > 0 and prefix[0] == '/':
            prefix = prefix[1:]
        files = [x for x in files if os.path.splitext(x)[-1] != '.pyc']

        src_to_copy.extend(
            [os.path.join(root, x) for x in files]
        )

        dst_to_copy.extend(
            [os.path.join(path_to_copy_folder, prefix, x) for x in files]
        )
    for (file_src, file_dst) in zip(src_to_copy, dst_to_copy):
        if not os.path.isdir(os.path.dirname(file_dst)):
            os.makedirs(os.path.dirname(file_dst))
        shutil.copy(file_src, file_dst)


def setup_experiment_folders(path_to_experiment_folder):
    for folder in [SRC_COPY_FOLDER, TB_LOG_FOLDER]:
        os.makedirs(os.path.join(path_to_experiment_folder, folder))


def safe_division(x, y): return 0.0 if float(y) == 0.0 else float(x) / float(y)


#def file_path_to_ID(path): return os.path.splitext(os.path.basename(path))[0]
def file_path_to_ID(path): return os.path.basename(path).split('.')[0]


def get_early_stopping_criterion(file_name): return (
    os.path.basename(file_name).split('best_on')[0]).split('.h5')[0]


def gauss_blur(image, sigma): return image if sigma == 0 else cv2.GaussianBlur(
    image, (0, 0), sigma)


def load_file_with_pre_processing(file_path, prep_fcn_image, prep_fcn_label):
    """Default function to load an image or label. Should be exchanged
    if the specific dataset has unique demands on how to load an image.

    Parameters
    ----------
    file_path : str
        path of the file that needs to be loaded
    prep_fcn : function
        pre-processing function that is applied after loading the image.
        If None, the image is not altered.
    Raises
    ------
    ValueError
        Unknown file type if the file's extension does not match '.jpg', 
        '.png', '.bmp' or '.npy'.

    Returns
    -------
    numpy array
        the loaded image or label.
    """
    file_ext = os.path.splitext(file_path)[-1]
    if file_ext in ['.jpg', '.png', '.bmp']:
        return load_image(file_path, prep_fcn_image)

    elif file_ext == '.npy':
        label = np.load(file_path)
        if prep_fcn_label is not None:
            return prep_fcn_label(label)
        else:
            return label

    else:
        raise ValueError('Unknown file type: {}'.format(file_ext))


if __name__ == "__main__":
    # _test_encoder()
    # test_mean_average_precision()

    test_empty_images()
