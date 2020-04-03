import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

# best practice: keep all constants in one config file
#from config import DATA_FOLDER, CIFAR10_RESIZE

DATA_FOLDER = '/data'  # change me
CIFAR10_RESIZE = 224  # so you can eval a imagenet pre-trained network


def get_dataloader(
    is_train=True, indeces=None, batch_size=32, num_workers=0,
        data_path=DATA_FOLDER):
    # indeces, if you only want to train a subset
    # https://discuss.pytorch.org/t/train-on-a-fraction-of-the-data-set/16743/6
    dataset = get_dataset(is_train, data_path)

    if indeces is None:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            sampler=SubsetRandomSampler(indeces)
        )
    return data_loader


def get_dataset(is_train=True, data_path=DATA_FOLDER):
    dataset = torchvision.datasets.CIFAR10(
        data_path,
        train=is_train,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((CIFAR10_RESIZE, CIFAR10_RESIZE)),
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
            torchvision.transforms.ToTensor(),

            torchvision.transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784]
            )

        ])
    )

    return dataset


if __name__ == "__main__":
    from DLBio.pytorch_helpers import cuda_to_numpy
    import matplotlib.pyplot as plt
    dataset = get_dataset()
    for x, y in dataset:
        x -= x.min()
        x /= x.max()
        plt.imshow((255. * cuda_to_numpy(x)).astype('uint8'))
        plt.show()
