import torch


def set_device(cuda: bool = True):
    """
    Set the device to cuda and default tensor types to FloatTensor on the device
    """
    # Set device
    device = torch.device("cuda" if (
        torch.cuda.is_available() and cuda) else "cpu")
    return device
