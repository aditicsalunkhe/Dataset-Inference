import torchvision
from torchvision import datasets, transforms
#from torch.util.data import DataLoader, Dataset, Subset

tr_normalize = transforms.Normalize((0.1307,), (0.3081,))

transform_train = transforms.Compose([transforms.ToTensor(), tr_normalize])
mnist_train = datasets.MNIST("data_new", train=True, download=True, transform=transform_train)
print(len(mnist_train))