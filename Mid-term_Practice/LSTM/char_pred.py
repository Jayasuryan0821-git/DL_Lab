import string
import torch
import torch.nn as nn

# Function to encode a character as a one-hot vector
def ltt(ch):    
    ans = torch.zeros(n_letters) 
    ans[letters.find(ch)] = 1   
    return ans

# Data preparation
data = "i love neural networks"
EOF = "#"
data = data.lower()
seq_len = len(data)
letters = string.ascii_lowercase + ' #'
n_letters = len(letters)

print('Letter set = ', letters, "len=", n_letters)
print("Encoding of 'a' ", ltt('a'))
print("Encoding of 'b' ", ltt('b'))
print("Encoding of '#' ", ltt('#'))

# Function to convert a sequence of characters into a tensor representation
def getLine(s):
    ans = []
    for c in s:
        ans.append(ltt(c))
    return torch.cat(ans, dim=0).view(len(s), 1, n_letters)

# LSTM model class
class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim)

    def forward(self, inp, hc):
        output, _ = self.LSTM(inp, hc)
        return output

# Initialize model, optimizer, and loss function
model = MyLSTM(n_letters, n_letters)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
LOSS = torch.nn.CrossEntropyLoss()

# Preparing targets for training
targets = [letters.find(x) for x in data[1:] + EOF]
targets = torch.tensor(targets)

# Preparing input tensor
inp = torch.cat([ltt(c) for c in data], dim=0).view(seq_len, 1, n_letters)

# Training loop
n_iters = 150
for itr in range(n_iters):
    model.zero_grad()
    h = torch.rand(1, 1, n_letters)
    c = torch.rand(1, 1, n_letters)
    output = model(inp, (h, c))
    output = output.view(seq_len, n_letters)
    loss = LOSS(output, targets)
    if itr % 10 == 0:
        print(itr, ' ', loss.item())
    loss.backward()
    optimizer.step()

# Function to make predictions
def predict(s):
    print("s= ", s)
    inp = getLine(s)
    h = torch.rand(1, 1, n_letters)
    c = torch.rand(1, 1, n_letters)
    out = model(inp, (h, c))
    predicted_index = out[-1][0].topk(1)[1].detach().numpy().item()
    predicted_letter = letters[predicted_index]
    print("Predicted letter:", predicted_letter)
    return predicted_letter

# Make a prediction
predict_str = "i love neu"
predict(predict_str)
