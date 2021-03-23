import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEBUG_PATCHWISE_SEGMENTATION = False
SHOW_IDX = 1


def whole_image_segmentation(model, image, fast_prediction=False, batch_size=4):

    input_shape = model.get_input_shape()
    output_shape = model.get_output_shape_for_patchwise_processing()

    original_image_size = {'x': image.shape[1], 'y': image.shape[0]}
    input_patch_size = {'x': input_shape[2], 'y': input_shape[1]}
    output_patch_size = {'x': output_shape[2], 'y': output_shape[1]}

    pad_x = input_patch_size['x'] - output_patch_size['x']
    pad_y = input_patch_size['y'] - output_patch_size['y']

    is_same_padding_network = (input_shape[1] == output_shape[1] and
                               input_shape[2] == output_shape[2])

    # same padding networks can suffer from border artifacts
    # rather predict overlapping segments of the image and cut the
    # borders.
    if is_same_padding_network:

        pad_x = 32
        pad_y = 32

        output_patch_size = {'x': output_patch_size['x'] - 2 * pad_x,
                             'y': output_patch_size['y'] - 2 * pad_y}

        def cnn_function(x, full_batch=False):
            out = model._predict(x, False, predict_patch=True)
            if out.ndim == 4 and not full_batch:
                out = out[0, pad_y:-pad_y, pad_x:-pad_x, :]
            elif out.ndim == 4 and full_batch:
                return out[:, pad_y:-pad_y, pad_x:-pad_x, :]
            else:
                return out[pad_y:-pad_y, pad_x:-pad_x, :]

    else:
        def cnn_function(x, full_batch=False):
            out = model._predict(x, False, predict_patch=True)
            if out.ndim == 4 and not full_batch:
                out = out[0, ...]
            elif out.ndim == 4 and full_batch:
                return out
            else:
                return out

    # if the original image size is smaller than the input of the network
    # the image is padded downward and upward and cut back to shape later on
    pad_right = max(0, input_patch_size['x'] - original_image_size['x'])
    pad_down = max(input_patch_size['y'] - original_image_size['y'], 0)
    padded_to_fit_input = False
    if pad_right > 0 or pad_down > 0:
        padded_to_fit_input = True
        image = np.lib.pad(
            image, ((0, pad_down), (0, pad_right), (0, 0)), 'symmetric')
        original_image_size = {'x': image.shape[1], 'y': image.shape[0]}

    output_full_size = {'x': image.shape[1], 'y': image.shape[0]}

    padding_size = {'x': pad_x, 'y': pad_y}

    image_padded = get_padded_image(image,
                                    padding=padding_size,
                                    pad_method='symmetric')

    output = patchwise_image_segmentation(
        cnn_function,
        image_padded,
        original_image_size,
        input_patch_size,
        output_patch_size,
        output_full_size,
        padding_size,
        model.get_num_classes(),
        use_fast=fast_prediction,
        batch_size=batch_size

    )

    # cutting back to original image size
    if padded_to_fit_input:
        if pad_down != 0:
            output = output[0:-pad_down, :, :]
        if pad_right != 0:
            output = output[:, 0:-pad_right, :]
    return output


def get_patch(padded_image, coordinates):
    x, y = coordinates['x'], coordinates['y']
    return padded_image[y[0]:y[1], x[0]:x[1]]


class ImagePatch(object):
    def __init__(self, **kwargs):
        self.input_patch = kwargs['input_patch']
        self.in_x = kwargs['in_x']
        self.in_y = kwargs['in_y']
        self.out_x = kwargs['out_x']
        self.out_y = kwargs['out_y']

        self.output = None


def _process(patches_, network_output_fcn, batch_size=4):
    processed_patches = []
    while patches_:
        tmp = []
        while len(tmp) < batch_size and patches_:
            tmp.append(patches_.pop())

        processed_patches += (__process_batch(tmp, network_output_fcn))

    return processed_patches


def _write(patches_, full_output):
    for patch in patches_:
        out_y = patch.out_y
        out_x = patch.out_x
        in_y = patch.in_y
        in_x = patch.in_x
        network_output = patch.output

        tmp = network_output[in_y[0]:in_y[1], in_x[0]:in_x[1], ...]
        full_output[out_y[0]:out_y[1], out_x[0]:out_x[1], ...] = tmp

    return full_output


def __process_batch(batch, network_output_fcn):
    input_batch = [x.input_patch for x in batch]
    input_batch = np.stack(input_batch, 0)
    output = network_output_fcn(input_batch, full_batch=True)

    for i in range(output.shape[0]):
        batch[i].output = output[i, ...]

    return batch


def patchwise_image_segmentation(network_output_fcn,
                                 padded_image,
                                 original_image_size,
                                 input_patch_size,
                                 output_patch_size,
                                 output_full_size,
                                 padding_size,
                                 num_classes,
                                 downsample_factor=1,
                                 network_input_fcn=get_patch,
                                 use_fast=True,
                                 batch_size=4
                                 ):
    """
      Function that processes a padded image patchwise and writes the
      output to an image with the shape given in output_full_size size.
      This function assumes that the middle of the input patch corresponds to
      the middle of the output patch.
      ----
      Arguments:
      network -> function returning the networks output tensor with dimensions
      (height, width, class) given an input patch of an image.

      padded_image -> image input for processing, padded if necessary
      ----
      Size arguments: dictionaries with 'x' and 'y' being width and height,
      respectively.

      original_image_size -> shape of the original, unpadded image

      input_patch_size ->  shape of the input patch that is given to the
      network

      output_patch_size -> shape of the network's returned output given a
      patch of size input_patch_size.

      output_full_size -> shape of the output once the whole image is processed

      padding_size -> specify the width/height of padding on the left and
      top border.
      ----
      4 Coordinate Frames are used:

      World Coordinates -> Frame of reference in floating point representation.
      All other coordinate frames can be accessed via this system. The origin
      is the middle of the upper left patches in the original and padded image.

      Input Coordinates -> Integer frame. Corresponding to the padded image.
      Padding size and Input_shape might not be identical. Therefore the
      padding vector (padding at one side e.g. left) is used to compute the
      coordinates of this frame.

      Output Coordinates -> Integer frame. Corresponding to the output image.

      Network Output Coordinates -> Integer frame. Corresponding to the network
      output patch.

      ######################################
      Coordinate transfer example:
      padding = 3
      output_size = 2
      input_size = 4
      0 -> position of input
            0 -> position of output
              0  -> init position of world coordinate
      |-|-|-|-|-|
      |pad--|out|
        |-input-|
          <---| .5 * input
      <---| + pad
      ->| .5*(input - output)
      #######################################
    """

    # ----------------define variables ------------------------------------------
    if downsample_factor > 1:
        print("Warning: downsample factor is {}.".format(downsample_factor))
    # downsampling is considered during the frame transformations
    output_patch_size['x'] = downsample_factor * output_patch_size['x']
    output_patch_size['y'] = downsample_factor * output_patch_size['y']

    # patch_center_world describes the current position in the process
    current_position_world = np.zeros(2).astype('float32')

    step_x = np.asarray([output_patch_size['x'], 0]).astype('float32')
    step_y = np.asarray([0, output_patch_size['y']]).astype('float32')

    # *directional vectors for input and output frame*
    # defined as distance from top-left of original to top_left of padded image
    distance_top_left_original_top_left_padded = np.asarray(
        [padding_size['x'],
         padding_size['y']]).astype('float32')

    # s: defined as distance from middle of patches to top_left of the respective patch
    distance_middle_top_left_output = .5 * np.asarray(
        [output_patch_size['x'],
         output_patch_size['y']]).astype('float32')

    distance_middle_top_left_input = .5 * np.asarray(
        [input_patch_size['x'],
         input_patch_size['y']]).astype('float32')

    offset = .5 * np.array([input_patch_size['x'] - output_patch_size['x'],
                            input_patch_size['y'] - output_patch_size['y']],
                           dtype='float32')

    # ------------define frame transformations-----------------------------------
    def world_2_input(v): return (
        v.astype('float32') +
        distance_top_left_original_top_left_padded +
        distance_middle_top_left_input -
        offset
    ).astype('int32')  # works for any padding

    def world_2_output(v): return (
        (v.astype('float32') +
         distance_middle_top_left_output) / downsample_factor).astype('int32')

    def output_2_world(v): return (
        v.astype('float32') * downsample_factor -
        distance_middle_top_left_output)

    def world_2_net(v): return (
        (v.astype('float32') -
         current_position_world +
         distance_middle_top_left_output) / downsample_factor).astype('int32')

    # define functions to move to the top left and down right pixel of the patches
    def top_left_input_world(): return current_position_world - \
        distance_middle_top_left_input

    def top_left_output_world(): return current_position_world - \
        distance_middle_top_left_output

    def down_right_input_world(): return current_position_world + \
        distance_middle_top_left_input

    def down_right_output_world(): return current_position_world + \
        distance_middle_top_left_output

    # allocate output image
    full_output = np.zeros(
        (output_full_size['y'], output_full_size['x'], num_classes))

    # compute number of steps so that every original pixel is processed
    num_steps_y = int(
        np.ceil(float(original_image_size['y']
                      ) / float(output_patch_size['y']))
    )
    num_steps_x = int(
        np.ceil(float(original_image_size['x']
                      ) / float(output_patch_size['x']))
    )
    assert num_steps_y > 0
    assert num_steps_x > 0
    # ------------- start of process -------------------------------------------
    patches_ = []
    for _i in range(num_steps_y):
        for _j in range(num_steps_x):

            # output
            top_left_output = world_2_output(top_left_output_world())
            down_right_output = world_2_output(down_right_output_world())

            # make sure that the current position creates a patch of size:
            # input_patch.shape
            is_changed = False
            if down_right_output[0] > original_image_size['x']:
                dx = down_right_output[0] - original_image_size['x']
                current_position_world -= np.array([dx, 0], dtype='float32')
                is_changed = True

            if down_right_output[1] > original_image_size['y']:
                dy = down_right_output[1] - original_image_size['y']
                current_position_world -= np.array([0, dy], dtype='float32')
                is_changed = True

            if is_changed:
                top_left_output = world_2_output(top_left_output_world())
                down_right_output = world_2_output(down_right_output_world())

            # input
            top_left_input = world_2_input(top_left_input_world())
            down_right_input = world_2_input(down_right_input_world())

            in_y = [top_left_input[1], down_right_input[1]]
            in_x = [top_left_input[0], down_right_input[0]]

            network_input = network_input_fcn(
                padded_image, {'x': in_x, 'y': in_y})
            if not use_fast:
                # case: is_single_image
                network_output = network_output_fcn(network_input)

            # update the net indeces accordingly
            top_left_world = output_2_world(top_left_output)
            down_right_world = output_2_world(down_right_output)

            top_left_network = world_2_net(top_left_world)
            down_right_network = world_2_net(down_right_world)

            # write to output image
            out_y = [top_left_output[1], down_right_output[1]]
            out_x = [top_left_output[0], down_right_output[0]]

            in_y = [top_left_network[1], down_right_network[1]]
            in_x = [top_left_network[0], down_right_network[0]]

            if use_fast:
                patches_.append(ImagePatch(
                    input_patch=network_input,
                    in_x=in_x,
                    in_y=in_y,
                    out_x=out_x,
                    out_y=out_y
                ))
            else:
                tmp = network_output[in_y[0]:in_y[1], in_x[0]:in_x[1], ...]
                full_output[out_y[0]:out_y[1], out_x[0]:out_x[1], ...] = tmp

            ####################### debug: show whats happening ###############
            if DEBUG_PATCHWISE_SEGMENTATION:
                _, ax = plt.subplots(1, 4)
                image_with_rectangle = np.copy(padded_image).astype('uint8')
                original_image = np.copy(
                    padded_image[padding_size['y']: padding_size['y'] +
                                 original_image_size['y'],
                                 padding_size['x']: padding_size['x'] +
                                 original_image_size['x']])
                top_left_rectangle = world_2_input(
                    output_2_world(top_left_output))
                down_right_rectangle = world_2_input(
                    output_2_world(down_right_output))
                cv2.rectangle(image_with_rectangle, tuple(
                    down_right_rectangle),
                    tuple(top_left_rectangle),
                    (255, 0, 0), 3)
                cv2.rectangle(image_with_rectangle, tuple(
                    top_left_input), tuple(down_right_input), (0, 0, 255), 3)
                cv2.circle(image_with_rectangle, tuple(
                    top_left_rectangle), 5, (0, 255, 0), -1)
                ax[0].imshow(image_with_rectangle)
                ax[0].set_title(
                    "Padded image. Red rectangle shows output, blue one input.\
                     Circle shows top_left of output.")
                ax[1].imshow(network_output[:, :, SHOW_IDX])
                ax[1].set_title("original output of the network")
                ax[2].imshow(full_output[:, :, SHOW_IDX])
                ax[2].set_title("full output")
                ax[3].imshow(original_image)
                ax[3].set_title("original image")
                plt.show()
                plt.close()
            ####################################################

            current_position_world += step_x

        current_position_world[0] = 0
        current_position_world += step_y

    if use_fast:
        patches_ = _process(patches_, network_output_fcn,
                            batch_size=batch_size)
        full_output = _write(patches_, full_output)

    return full_output


def get_padded_image(original_image,
                     patch_size=-1,
                     padding={'x': 0, 'y': 0},
                     pad_method='symmetric'):
    """
      Returns a padded image as a pre processing step for the network
      segmentation. For fully convolutional networks the patch_size
      (size of the examples it was trained with) needs to be specified. The
      image is then padded by half the patch size in each direction. If an
      end-to-end-trained network is used the padding is specified by the
      padding dictionary.
    """
    pad_with_patch_size = patch_size != -1

    if pad_with_patch_size:
        # this is the size of the edges around the image
        half_ps = patch_size // 2
        pad_x, pad_y = half_ps, half_ps
    else:
        pad_x, pad_y = padding['x'], padding['y']

    if original_image.ndim == 2:
        original_image = original_image[:, :, np.newaxis]

    if pad_method == "symmetric":
        padded_image = np.lib.pad(original_image,
                                  ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
                                  pad_method
                                  )
    elif pad_method == "constant":
        padded_image = np.lib.pad(original_image,
                                  ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
                                  pad_method,
                                  **{'constant_values': (
                                     (255, 255),
                                      (255, 255),
                                      (255, 255))
                                     }
                                  )

    return padded_image

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _test_stitching():
    import random
    D = 1024

    for _ in range(100):
        # works if padding < in_shape
        in_shape = [2 * random.randint(32, D // 2) for _ in range(3)]
        out_shape = in_shape
        model = FakeModel(in_shape, out_shape)
        image = np.random.randint(low=0, high=255, size=(D, D, 3))

        out = whole_image_segmentation(model, image, fast_prediction=True)

        if ((out - image)**2.).sum() > 1e-9:
            _, ax = plt.subplots(1, 3)
            ax[0].imshow(image)
            ax[1].imshow(out)
            ax[2].imshow(image - out)
            plt.savefig('debug.png')
            plt.close()
            assert ((out - image)**2.).sum() < 1e-9

    print('Test succeeded')


class FakeModel():
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def get_input_shape(self):
        return self.in_shape

    def get_output_shape_for_patchwise_processing(self):
        return self.out_shape

    def get_num_classes(self):
        return 3

    def _predict(self, x, *args, **kwargs):
        return x


if __name__ == "__main__":
    _test_stitching()
