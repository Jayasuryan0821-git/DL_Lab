import torch
import torch.nn as nn

# Define your input data
data = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print("Data: ", data.shape, "\n\n", data)

# Define hyperparameters
INPUT_SIZE = 1
SEQ_LENGTH = 5
HIDDEN_SIZE = 1
NUM_LAYERS = 1
BATCH_SIZE = 4

# Initialize your RNN model
rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)

# Initialize weights and biases
rnn.weight_ih_l0.data.fill_(1)
rnn.weight_hh_l0.data.fill_(1)
rnn.bias_ih_l0.data.fill_(1)
rnn.bias_hh_l0.data.fill_(1)

# Reshape your input data to fit the RNN input shape
inputs = data.view(BATCH_SIZE, -1, INPUT_SIZE)

# Pass the input through the RNN
out, h_n = rnn(inputs)

# Print shapes and results
print('Input: ', inputs.shape, '\n', inputs)
print('\nOutput:', out.shape, '\n', out)
print('\nHidden: ', h_n.shape, '\n', h_n)
