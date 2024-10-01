from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy
import random
import optuna

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

num_workers = multiprocessing.cpu_count()

# --------------------------------------- Define the model -------------------------------#
class CNN(nn.Module):
    def __init__(self, dropout_rate, fc1_neurons):
        super(CNN, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # Fully connected layers
        self.fc1 = nn.Linear(320, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 47)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten data
        x = x.view(-1, 320) #x.view(x.size(0), -1) # x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)# x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scaler = GradScaler()

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss / len(test_loader.dataset), accuracy

# ------------------------- Hyperparameter tuning -------------------------------------#
# Objective function for Optuna
def objective(trial):
    # Suggest values for hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 1.0)
    fc1_neurons = trial.suggest_int('fc1_neurons', 50, 200)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(dropout_rate, fc1_neurons).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training loop
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, criterion)
        _, accuracy = test(model, device, test_loader, criterion)
    
    # Optuna minimizes the returned value, so we return -accuracy to maximize it
    return -accuracy

# Create a study to optimize the objective
#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=20)  # Run 50 trials of optimization

# Get the best trial and hyperparameters
#print("Best trial:")
#trial = study.best_trial

#print(f"  Value: {trial.value}")
#print("  Hyperparameters: ")
#for key, value in trial.params.items():
    #print(f"    {key}: {value}")
# ------------------------------End of Hyperparameter tuning-------------------------------#

# ---------------------------------- Optimal model parameters ---------------------------#
# Create data loaders
loaders = {
        'train': DataLoader(train_data, batch_size=256, shuffle=True, num_workers=num_workers),
        'test': DataLoader(test_data, batch_size=256, shuffle=True, num_workers=num_workers),
        }

# (dropout_rate, fc1_neurons)
learning_rate = 0.0004699598617965072
dropout_rate = 0.33965070330128894

model = CNN(dropout_rate, 196).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 16):
    train(model, device, loaders['train'], optimizer, loss_fn)
    test(model, device, loaders['test'], loss_fn)

# Save the model 
torch.save(model.state_dict(), './character_recognition.torch')
print("Model saved.")
# Load the model 
# model.load_state_dict(torch.load('./character_recognition.torch'))

decoded_targets = [decode_emnist(target) for _, target in test_data]

def predict_images(attempt=100):
    pass
    model.eval()
    for i in range(attempt):
        rand_idx = random.randint(0, len(test_data) - 1)
        data, _ = test_data[rand_idx]
        data = data.unsqueeze(0).to(device)
        output = model(data)
        pred_idx = output.argmax(dim=1, keepdim=True).item()
        prediction = decode_emnist(pred_idx)
        actual = decoded_targets[rand_idx]
        print(f"Prediction: {prediction}, Actual: {actual}")
        image = data.squeeze(0).squeeze(0).cpu().numpy()
        plt.imshow(image, cmap='gray')
        plt.show()

predict_images(100)
print(device)
