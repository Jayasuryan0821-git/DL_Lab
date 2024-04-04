import torch
from torch import nn
import torchvision 
from torchvision import datasets,models,transforms
from torchvision.models import alexnet,AlexNet_Weights
from torch.utils.data import dataloader
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self, input_shape,hidden_units,output_shape):
        super().__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.blk2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,out_features=output_shape)
        )

    def forward(self,x):
        x = self.blk1(x)
       # print(f"blk1 shape : {x.shape}")
        x = self.blk2(x)
       # print(f"blk2 shape : {x.shape}")
        x = self.clf(x)
        #print(f"clf shape : {x.shape}")
        return x

train_data = datasets.MNIST(root='data/',download=True,train=True,target_transform=None,transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data/',download=True,train=False,target_transform=None,transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=32,shuffle=False)
# img,label = train_data[0]
# print(img.shape)
# train_features_batch,train_labels_batch = next(iter(train_loader))
# print(train_features_batch.shape)
def train_step(model,train_loader,loss_fn,optimizer,device):
    model.train()
    train_loss,train_acc = 0,0
    for batch,(X,y) in enumerate(train_loader):
        X,y =  X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (torch.eq(y,y_pred.argmax(dim=1)).sum().item()/ len(y_pred))*100
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print(f"Train loss : {train_loss:.4f} | Train acc : {train_acc:.4f}")

def test_step(model,test_loader,loss_fn,device):
    test_loss,test_acc = 0,0
    model.eval()
    for batch,(X,y) in enumerate(test_loader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        test_acc += (torch.eq(y_pred.argmax(dim=1),y).sum().item() / len(y_pred))*100
        test_loss += loss.item()
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss : {test_loss:.4f} | Test acc: {test_acc:.4f}")

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    train_step(model,train_loader,loss_fn,optimizer,device)
    test_step(model,test_loader,loss_fn,device)num_classes = len(train_data.classes)
model = CNN(input_shape=1,hidden_units=10,output_shape=num_classes).to(device)
# rand_img_tensor = torch.rand(size=(1,28,28))
# print(rand_img_tensor.shape) 
# rand_img_tensor = rand_img_tensor.unsqueeze(dim=0)
# print(rand_img_tensor.shape)
# model(rand_img_tensor.to(device))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

def train_step(model,train_loader,loss_fn,optimizer,device):
    model.train()
    train_loss,train_acc = 0,0
    for batch,(X,y) in enumerate(train_loader):
        X,y =  X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (torch.eq(y,y_pred.argmax(dim=1)).sum().item()/ len(y_pred))*100
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print(f"Train loss : {train_loss:.4f} | Train acc : {train_acc:.4f}")

def test_step(model,test_loader,loss_fn,device):
    test_loss,test_acc = 0,0
    model.eval()
    for batch,(X,y) in enumerate(test_loader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        test_acc += (torch.eq(y_pred.argmax(dim=1),y).sum().item() / len(y_pred))*100
        test_loss += loss.item()
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss : {test_loss:.4f} | Test acc: {test_acc:.4f}")

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    train_step(model,train_loader,loss_fn,optimizer,device)
    test_step(model,test_loader,loss_fn,device)

from pathlib import Path 
model_path = Path('model_list')
model_path.mkdir(parents=True,exist_ok=True)
model_name = 'cnn.pth'
model_save_path = model_path / model_name 
torch.save(obj=model.state_dict(),f=model_save_path)
