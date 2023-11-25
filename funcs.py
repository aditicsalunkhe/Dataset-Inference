import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset


def get_dataloaders(dataset, batch_size, normalize = False, train_shuffle = True):
    if dataset == "CIFAR10":
        data_source = datasets.CIFAR10
        tr_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) if normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), tr_normalize,
                                    transforms.Lambda(lambda x: x.float())])
        transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])
        if not train_shuffle:
            transform_train = transform_test
        
        train = data_source("data", train=True, download=True, transform=transform_train)
        test = data_source("data", train=False, download=True, transform=transform_test)
        train_subset = Subset(train, range(0, 3000))
        test_subset = Subset(test, range(0, 750))
        train_loader = DataLoader(train_subset, batch_size = batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(test_subset, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader