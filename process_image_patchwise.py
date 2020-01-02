import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEBUG_PATCHWISE_SEGMENTATION = False
SHOW_IDX = 1


def whole_image_segmentation(model, image):

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

        def cnn_function(x):
            return model._predict(
                x, False, predict_patch=True)[pad_y:-pad_y, pad_x:-pad_x, :]

    else:
        def cnn_function(x): return model._predict(
            x, False, predict_patch=True)

    # if the original image size is smaller than the input of the network
    # the image is padded downwart and upward and cut back to shape later on
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

    output = patchwise_image_segmentation(cnn_function,
                                          image_padded,
                                          original_image_size,
                                          input_patch_size,
                                          output_patch_size,
                                          output_full_size,
                                          padding_size,
                                          model.get_num_classes())

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


def patchwise_image_segmentation(network_output_fcn,
                                 padded_image,
                                 original_image_size,
                                 input_patch_size,
                                 output_patch_size,
                                 output_full_size,
                                 padding_size,
                                 num_classes,
                                 downsample_factor=1,
                                 network_input_fcn=get_patch
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

      Network Output Coordinages -> Integer frame. Correspondig to the network
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

    # ------------- start of process -------------------------------------------
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

            full_output[out_y[0]:out_y[1], out_x[0]:out_x[1], ...
                        ] = network_output[in_y[0]:in_y[1], in_x[0]:in_x[1],
                                           ...]

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
