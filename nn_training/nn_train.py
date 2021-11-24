from torch.nn.modules.activation import ReLU
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np

class FontDataset(Dataset):

    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        label = np.zeros(62)
        label[self.labels[index]-1] = 1
        return self.transform(self.data[index]), torch.from_numpy(label)
    

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(64),
        T.ToTensor()])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64,2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048, 62)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    data_path = ".\\fnt_data.npy"
    labels_path = ".\\fnt_data_labels.npy"
    test_path = ".\\fnt_test_data.npy"
    test_labels_path = ".\\fnt_test_labels.npy"

    print("Loading data...")
    X_train = np.load(data_path)
    Y_train = np.load(labels_path)
    X_test = np.load(test_path)
    Y_test = np.load(test_labels_path)
    print(f"Shape of input data: {X_train.shape}")
    print(f"Data type: {type(X_train[0][0][0])}")
    print(f"Shape of labels: {Y_train.shape}")
    print(f"Labels type: {type(Y_train[0])}")


    train_data = FontDataset(X_train, Y_train)
    test_data = FontDataset(X_test, Y_test)

    batch_size = 64

    print("Creating loader...")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=3)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=3)
    print("Initiating model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model1.pth")
    
