import os
import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Define dataset directory
base_dir = Path('data/dataset/names')
file_paths = list(base_dir.glob('*'))

# Initialize variables
all_names = []
labels = []
categories = [file_path.stem for file_path in file_paths]
num_classes = len(categories)
max_len_name = 0

# Determine the maximum name length
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        names = f.read().split("\n")
        for name in names:
            if len(name) > max_len_name:
                max_len_name = len(name)

# Prepare the data
for i, file_path in enumerate(file_paths):
    nationality_index = categories.index(file_path.stem)
    with open(file_path, 'r', encoding='utf-8') as f:
        names = [name for name in f.read().split('\n') if name]  # Exclude empty names
    for name in names:
        num_list = [ord(k) for k in name] + [0] * (max_len_name - len(name))
        all_names.append(num_list)
        labels.append(nationality_index)

# Define dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

# Define RNN model
class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, X):
        output, _ = self.rnn(X)
        output = output[:, -1, :]
        output = self.fc1(output)
        return output

# Create dataset and data loader
data = MyDataset(all_names, labels)
data_loader = DataLoader(data, batch_size=32, shuffle=True)

# Initialize model, loss criterion, and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RNN_Model(input_size=1, hidden_size=12, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
epochs = 5
for epoch in range(epochs):
    train_loss = 0 
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        X = X.view(-1, max_len_name, 1)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(data_loader)
    print(f"Epoch {epoch+1} | Loss: {train_loss}")

def predict(model,name,max_len_name,categories,device):
    name_ascii = [ord(char) for char in name] + [0] * (max_len_name - len(name))
    name_tensor = torch.tensor([name_ascii],dtype=torch.float32).view(-1,max_len_name,1).to(device)
    model.eval()
    with torch.inference_mode():
        output = model(name_tensor)
    _,predicted_index = torch.max(output,1)
    return categories[predicted_index.item()]

sample_input = 'Alexander'
sample_out =  predict(model,sample_input,max_len_name,categories,device)
print(f"The predicted nationality for {sample_input} is {sample_out}.")
