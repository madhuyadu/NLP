import torch
import torch.nn as nn


class NeuralNet(nn.Module):  #inherits from nn.Module class
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)   # input layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # hidden layer 
        self.l3 = nn.Linear(hidden_size, num_classes)  # output layer
        self.relu = nn.ReLU()                          # activation function, same across all layers
    

    # Create forward pass
    def forward(self, x):
        out = self.l1(x)  
        out = self.relu(out)  # output of 1st layer
        out = self.l2(out)
        out = self.relu(out)  # output of 2nd layer
        out = self.l3(out)    # output of final layer
        # no activation and no softmax at the end --> this will be done later in a different section
        return out