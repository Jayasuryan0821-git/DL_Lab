import os
import torch
from torch import nn
from pathlib import Path
import glob
from torch.utils.data import DataLoader, Dataset

# Assuming num_classes is the total number of nationalities in your dataset
num_classes = 0  # Replace 0 with the actual number of nationalities

labels = []
all_names = []  # Use a different variable to store the lists of names
base_dir = Path('data/dataset/names')
files = list(base_dir.glob('*'))

# Iterate over each nationality file
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        names = f.read()
    names_list = names.split('\n')

    # Process each name in the file
    for name in names_list:
        num_list = [ord(k) for k in name] + [0] * (20 - len(name))  # Pad to ensure length is 20
        all_names.append(num_list)

    labels.extend([num_classes] * len(names_list))
    num_classes += 1

class RNN(nn.Module):
    def __init__(self, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=12, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(12, num_classes)  # Adjusted to have a neuron for each class

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:, -1, :]  # Get the last time step output
        output = self.fc1(output)
        return output

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

data = MyDataset(all_names, labels)
train_loader = DataLoader(dataset=data, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RNN(num_classes).to(device)
epochs = 500
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(epochs):
    running_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X.view(-1, 20, 1))  # Ensure input is correctly shaped
        loss = criterion(y_pred, y.long())  # Ensure target is a long tensor
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {running_loss}")
