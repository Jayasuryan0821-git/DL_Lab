import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader

text = ['hey how are you', 'good i am fine', 'have a nice day']
all_chars = set()
next_chars, sequences = [], []
max_seq_len = 10

# Update all_chars based on the provided text
for sentence in text:
    all_chars.update(sentence)
all_chars = sorted(list(all_chars))
num_chars = len(all_chars)
char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
index_to_char = {i: ch for i, ch in enumerate(all_chars)}

# Generate sequences and next_chars from the text variable
for sentence in text:
    for i in range(len(sentence) - 1):
        start_index = max(0, i + 1 - max_seq_len)
        end_index = i + 1
        sequence = [char_to_idx[ch] for ch in sentence[start_index:end_index]]
        sequence += [0] * (max_seq_len - len(sequence))  # Pad the sequence if necessary
        sequences.append(sequence)
        next_chars.append(char_to_idx[sentence[i + 1]])

class Data(Dataset):
    def __init__(self, sequences, next_chars):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.next_chars = torch.tensor(next_chars, dtype=torch.long)
    
    def __getitem__(self, index):
        return self.sequences[index], self.next_chars[index]
    
    def __len__(self):
        return len(self.sequences)
    
dataset = Data(sequences, next_chars)
data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

class RNN_Model(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=num_chars, embedding_dim=hidden_size)
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_chars)
        
    def forward(self, x):
        x = self.embeddings(x)
        output, _ = self.rnn(x)
        output = output[:, -1, :]
        output = self.fc(output)
        return output
    
criterion = nn.CrossEntropyLoss()
model = RNN_Model(num_chars, hidden_size=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for seq, next_char in data_loader:
        seq, next_char = seq.to(device), next_char.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, next_char)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
