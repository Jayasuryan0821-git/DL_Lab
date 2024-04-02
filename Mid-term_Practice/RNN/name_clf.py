import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader

# Define your dataset here
dataset_dict = {
    "Category1": ["Name1", "Name2"],
    "Category2": ["Name3", "Name4"],
    # Add more categories and names as needed
}

categories = list(dataset_dict.keys())
num_classes = len(categories)
all_names, labels = [], []
max_len = max(max(len(name) for name in names) for names in dataset_dict.values())

# Process the dataset
for category_index, (category, names) in enumerate(dataset_dict.items()):
    for name in names:
        num_list = [ord(char) for char in name] + [0] * (max_len - len(name))
        all_names.append(num_list)
        labels.append(category_index)

print(len(all_names), len(labels))

class Data(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

class RNN_Char(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, X):
        output, _ = self.rnn(X)
        output = output[:, -1, :]
        output = self.fc(output)
        return output

dataset = Data(all_names, labels)
data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RNN_Char(input_size=1, hidden_size=10, output_size=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
epochs = 5
for epoch in range(epochs):
    train_loss = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        X = X.view(-1, max_len, 1)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(data_loader)
    print(f"Epoch: {epoch + 1} | Train loss: {train_loss}")

# Prediction function remains mostly unchanged
def predict(model, categories, name, max_len, device):
    name_ascii = [ord(char) for char in name] + [0] * (max_len - len(name))
    name_tensor = torch.tensor([name_ascii], dtype=torch.float32)
    name_tensor = name_tensor.view(-1, max_len, 1).to(device)
    with torch.no_grad():
        y_pred = model(name_tensor)
        pred_idx = y_pred.argmax(dim=1)
    nationality_index = categories[pred_idx.item()]
    return nationality_index

# Example usage
sample_input = 'Alexander'
sample_out = predict(model, categories, sample_input, max_len, device)
print(f"The predicted nationality for {sample_input} is {sample_out}.")
