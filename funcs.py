import torchvision
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset



def get_dataloaders(dataset, batch_size, student_or_teacher):
    # For any new dataset that you are using, add the normalize mean and standard deviation parameters for the dataset
    if dataset == "MNIST":
        tr_normalize = transforms.Normalize((0.1307,), (0.3081,))
    
    transform_train = transforms.Compose([transforms.ToTensor(), tr_normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize]) 
    mnist_train = datasets.MNIST("data", train=True, download=True, transform=transform_train)
    mnist_test = datasets.MNIST("data", train=False, download=True, transform=transform_test)
    if student_or_teacher == 'student':
        mnist_train = Subset(mnist_train, range(len(mnist_train)//4))
        mnist_test = Subset(mnist_test, range(len(mnist_test)//4))
    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

def load(model, model_name):
    try:
        model.load_state_dict(torch.load(f"{model_name}.pt"))
    except:
        dictionary = torch.load(f"{model_name}.pt")['state_dict']
        new_dict = {}
        for key in dictionary.keys():
            new_key = key[7:]
            if new_key.split(".")[0] == "sub_block1":
                continue
            new_dict[new_key] = dictionary[key]
        model.load_state_dict(new_dict)
    return model


def epoch_test(args, loader, model, stop = False):
    """Evaluation epoch over the dataset"""
    test_loss = 0; test_acc = 0; test_n = 0
    func = lambda x:x
    with torch.no_grad():
        for batch in func(loader):
            X,y = batch[0].to('cpu'), batch[1].to('cpu')
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            test_loss += loss.item()*y.size(0)
            test_acc += (yp.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            if stop:
                break
    return test_loss / test_n, test_acc / test_n        