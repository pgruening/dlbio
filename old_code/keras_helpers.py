import keras.backend as K


def check_if_is_init(tensor):
    """When using costum functions that slice inputs,
    there can be problems in the first run is the input
    has a None somewhere

    Parameters
    ----------
    tensor : tensor
        input tensor like y_true or y_pred
    Returns
    -------
    boolean
        True if tensor shape contains a None value
    """
    shape = K.int_shape(tensor)
    for s in shape:
        if s is None:
            return True
    return False
