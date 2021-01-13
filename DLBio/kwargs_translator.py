"""
    You can use the kwargs translator to pass different arguments, which is
    helpful if input arguments for a module 
    ("python my_module.py --argument1 value1 --argument2") change often.
    For example, training different CNN architectures: each architecture
    has other input arguments to define the specific CNN. 

    """


def get_kwargs(in_str):
    """Transform a keyword argument string into a dictionary

    Parameters
    ----------
    in_str : str
        keyword argument string, is computed in to_kwarg_str

    Returns
    -------
    dictionary
        Values are always lists. Thus, even single arguments need a 
        single_value = dict[key][0] to grab them.
        None and Boolean values are type-casted.
        Any other arguments are strings and need to be casted elsewhere. E.g.:
        single_value = int(dict[key][0]).
    """
    if in_str is None:
        return dict()
    # key1: value1 value2 value3; key2: value ...
    splits_ = in_str.split(';')
    out = dict()
    for split in splits_:
        if split == '':
            continue

        tmp = split.split(':')
        key = tmp[0]
        val = tmp[1]

        # remove first space
        tmp = val.split(' ')[1:]
        out[key] = []
        for x in tmp:
            if x == 'None':
                x = None
            if x == 'True':
                x = True
            if x == 'False':
                x = False

            out[key].append(x)

    return out


def to_kwargs_str(in_dict):
    """Transforms a dictionary into a string that can be transformed back
    via get_kwargs. All values need to be lists.

    Parameters
    ----------
    in_dict : dictionary

    Returns
    -------
    str
        Keys and values are separated by colon+space ": " .
        List items are separated by spaces " " .
        Key-value pairs are separated by semicolons ";" .
        Example: 
        input: {key1: [value1, value2, value3], key2: [value]}
        output: "key1: value1 value2 value3; key2: value"
    """
    out = ''
    for key, val in in_dict.items():
        out += f'{key}:'

        assert isinstance(
            val, list), f'input value for {key} needs to be a list'
        for v in val:
            out += f' {v}'

        out += ';'

    return out


def _debug():
    X = {
        'a': [-1],
        'b': [1, 3, 5],
        'c': ['test'],
        'd': [None],
    }

    Y = to_kwargs_str(X)
    Z = get_kwargs(Y)

    print(X)
    print(Z)

    xxx = 0


if __name__ == "__main__":
    _debug()
