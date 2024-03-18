import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_features):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear_layer = nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self, x):
        # Forward pass through the RNN
        output, hidden_state = self.rnn(x)
        # Forward pass through the linear layer (hidden-to-output)
        hidden_to_output = self.linear_layer(hidden_state[-1])
        return output, hidden_state, hidden_to_output

# Example input tensor
Input_size = 4
Hidden_size = 3
Num_layers = 1
Batch_size = 1
Sequence_length = 4
Out_features = 4
x = torch.ones((Batch_size, Sequence_length, Input_size))
print("x=", x)

# Instantiate the SimpleRNN model
model = SimpleRNN(Input_size, Hidden_size, Num_layers, Out_features)

# Initialize weights and biases to 1 for both RNN and linear layer
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.ones_(param.data)
    elif 'bias' in name:
        nn.init.ones_(param.data)
