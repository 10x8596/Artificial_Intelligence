from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy
import random

emnist_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

def decode_emnist(pred_idx):
    return emnist_mapping[pred_idx]

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

# Load the dataset
train_data = datasets.EMNIST(
    root='./data', split='balanced', train=True,
    download=True, transform=transform
)
test_data = datasets.EMNIST(
    root='./data', split='balanced', train=False,
    download=True, transform=transform
)

# Create data loaders
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
loaders = {
        'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
        }

# Create model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # Fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten data
        x = x.view(x.size(0), -1) # x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\t {loss.item():.6f}")

def test():
    model.eval()
    test_loss = 0 
    correct = 0 

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test']) # ['test'].dataset
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%\n)")

for epoch in range(1, 11):
    train(epoch)
    test()

# Save the model 
torch.save(model.state_dict(), './character_recognition.torch')
print("Model saved.")
# Load the model 
# model.load_state_dict(torch.load('./character_recognition.torch'))

def predict_images(attempt=100):
    model.eval()
    for i in range(attempt):
        rand_idx = random.randint(0, len(test_data) - 1)
        data, target = test_data[rand_idx]
        data = data.unsqueeze(0).to(device)
        output = model(data)
        pred_idx = output.argmax(dim=1, keepdim=True).item()
        prediction = decode_emnist(pred_idx)
        actual = decode_emnist(target)
        print(f"Prediction: {prediction}, Actual: {actual}")
        image = data.squeeze(0).squeeze(0).cpu().numpy()
        plt.imshow(image, cmap='gray')
        plt.show()

predict_images(100)
print(device)
