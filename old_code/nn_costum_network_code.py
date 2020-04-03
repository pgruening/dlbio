import keras.backend as K
import keras


def get_offset_layer(same_level_input, upsampled_input):
    """If the two input layers do not match, return a cropping layer.

    Parameters
    ----------
    same_level_input : keras layer
      Bigger layer coming from the left side of the unet
    upsampled_input : keras layer
      Smaller layer coming from a deeper level and being upsampled
    Returns
    -------
    Boolean and keras layer
      If a cropping layer is needed, True is returned with the layer.
      This layer is added behind same_level_input.
      Else, False and None is returned.
    """
    # offset layers
    # NOTE: used _keras_shape before
    if K.int_shape(same_level_input)[2] == K.int_shape(upsampled_input)[2]:
        offset_x = 0
    else:
        offset_x = K.int_shape(same_level_input)[2] - \
            K.int_shape(upsampled_input)[2]

    if K.int_shape(same_level_input)[1] == K.int_shape(upsampled_input)[1]:
        offset_y = 0
    else:
        offset_y = K.int_shape(same_level_input)[1] - \
            K.int_shape(upsampled_input)[1]

    if offset_x > 0 or offset_y > 0:

        offset_x_left = offset_x//2
        offset_x_right = offset_x - offset_x_left

        offset_y_left = offset_y//2
        offset_y_right = offset_y - offset_y_left

        """
        same_level_crop = keras.layers.Cropping2D(
            cropping=((offset_y_left, offset_y_right),
                      (offset_x_left, offset_x_right))
        )(same_level_input)
        """
        same_level_crop = keras.layers.Cropping2D(
            cropping=((offset_y//2, offset_x//2))
        )(same_level_input)
        return True, same_level_crop
    else:
        return False, None
