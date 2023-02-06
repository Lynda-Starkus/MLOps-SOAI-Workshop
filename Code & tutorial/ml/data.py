import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def load_mnist_data(root='data', flatten=True, batch_size=32):
    if flatten:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
        )
    else:
        transform = torchvision.transforms.ToTensor(),

    train_dataset = MNIST(root=root, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False,
                         download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader



    """_summary_
    
        The script starts by importing the torch and torchvision libraries, as well as the MNIST dataset from the torchvision.datasets module 
        and the DataLoader class from the torch.utils.data module.

        The load_mnist_data function takes three parameters: root, flatten, and batch_size. The root parameter specifies the directory 
        where the MNIST data should be stored, flatten is a flag indicating whether the images should be flattened into a 1D tensor, 
        and batch_size specifies the number of samples to be included in each batch of data.

        The function first creates a transform object, which is a composition of two functions:

        torchvision.transforms.ToTensor(), which converts the image data from a PIL image to a PyTorch tensor, and
        torchvision.transforms.Lambda(lambda x: torch.flatten(x)), which flattens the image into a 1D tensor if the flatten flag is set to True.
        If the flatten flag is False, only the ToTensor() function is used.
        
        The script then creates two datasets, train_dataset and test_dataset, using the MNIST class, with the root directory and 
        transform specified. The train argument of the MNIST class is set to False for the test dataset, meaning it will only load the test data.

        The script then creates two data loaders, train_dataloader and test_dataloader, using the DataLoader class, specifying 
        the respective datasets and the batch size. The shuffle argument is set to True for the training data loader, meaning that 
        the order of the training data will be shuffled after each epoch.

        Finally, the function returns the two data loaders.
    
    """