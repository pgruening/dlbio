def cuda_to_numpy(x):
    x = x.cpu().detach()
    if x.dim() == 4:
        return x.permute([0, 2, 3, 1]).numpy()
    elif x.dim() == 3:
        return x.permute([1, 2, 0]).numpy()
    else:
        raise ValueError('Unkwnown dimensionality: {}'.format(x.dim()))
