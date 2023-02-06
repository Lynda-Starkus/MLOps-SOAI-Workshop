import torch
import torchvision


class LinearModel(torch.nn.Module):
    def __init__(self, hyperparameters: dict):
        super(LinearModel, self).__init__()

        # Get model config
        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']
        self.hidden_dims = hyperparameters['hidden_dims']
        self.negative_slope = hyperparameters.get("negative_slope", .2)

        # Create layer list
        self.layers = torch.nn.ModuleList([])
        all_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))

        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = torch.nn.functional.leaky_relu(
                x, negative_slope=self.negative_slope)
        x = self.layers[-1](x)
        return torch.nn.functional.softmax(x, dim=-1)


        """_summary_
        
        This is a PyTorch implementation of a simple linear neural network model. It's a subclass of torch.nn.Module, 
        which is the base class for all neural network modules in PyTorch. The class definition includes an init method, 
        which is called when the model is instantiated, and a forward method, which defines the forward pass of the network.

        The init method takes a dictionary hyperparameters as an argument, which is used to configure the model. 
        The method uses the input_dim, output_dim, hidden_dims and negative_slope hyperparameters to configure the architecture of the model.

        The model consists of a number of linear layers, as determined by the number of hidden dimensions, 
        and each layer is followed by a leaky ReLU activation function. The final layer is followed by a 
        softmax activation to produce a probability distribution over the possible output classes.

        The forward method implements the forward pass of the network, which takes an input tensor x and passes it through each layer of 
        the network, applying the activation functions after each layer. The final output is the predicted probability 
        distribution over the possible output classes.

        """
        