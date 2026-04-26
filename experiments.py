import warnings

# catch everything from numpy / torchvision CIFAR loader
warnings.filterwarnings(
    "ignore",
    message=".*align should be passed as Python or NumPy boolean.*"
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import models
from models.PredictorHyperNet import PredictorHyperNet

# =========================
# Config
# =========================
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Data
# =========================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

trainset = torchvision.datasets.CIFAR100(root="./data", train=True,
                                         download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                        download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=256,
                        shuffle=False, num_workers=4)


model = PredictorHyperNet().to(DEVICE)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()

base_params = []
trunk_params = []
head_params = []
gate_params = []

for name, param in model.hypernet.named_parameters():
    if "hyper_trunk" in name:
        trunk_params.append(param)
    elif "delta_weight_head" in name:
        head_params.append(param)
    elif "gate_head" in name:
        gate_params.append(param)
    else:
        base_params.append(param)

optimizer = torch.optim.SGD(
    [
        {"params": model.predictor.parameters(), "lr": 0.05, "weight_decay": 5e-4},
        {"params": base_params,  "lr": 0.05,  "weight_decay": 5e-4},
        {"params": trunk_params, "lr": 0.05, "weight_decay": 5e-4},
        {"params": head_params,  "lr": 0.05, "weight_decay": 5e-4},  # slightly lower
    ],
    momentum=0.9,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# Training
# =========================
def train(epoch):


    model.train()
    total_loss = 0
    correct = 0
    total = 0

    if model.freeze_predictor:
        model.predictor.eval()

    for i, (inputs, targets) in enumerate(trainloader):

        if epoch == 1:
            print(i)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()


        outputs, priors = model(inputs)
        loss = criterion(outputs, targets)# F.cross_entropy(priors, targets, label_smoothing=0.1)

        if not model.freeze_predictor:
            loss += F.cross_entropy(priors, targets, label_smoothing=0.1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch {epoch}: Train Loss {total_loss:.3f} | Acc {100.*correct/total:.2f}%")

# =========================
# Evaluation
# =========================
def test(mult = 1):
    model.eval()
    correct = 0
    correct2 = 0
    
    total = 0



    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            inputs = inputs * mult

            outputs, priors = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            _, predicted = priors.max(1)
            correct2 += predicted.eq(targets).sum().item()

    print(f"Test Acc: {100.*correct/total:.2f}%, Priors Acc: {100.*correct2/total:.2f}%")


# =========================
# Main
# =========================
if __name__ == "__main__":

    print(DEVICE)

    for epoch in range(1, EPOCHS + 1):

        # train and test with full data
        model.zero_input = False
        model.no_prior = False
        train(epoch)
        test()

        # testing hypernetwork bypass
        
        model.zero_input = True # set image to 0
        model.no_prior = False 
        test()
        model.zero_input = False
        model.no_prior = True # set prior prediction to 0
        test()
        

        scheduler.step()

    torch.save(model.state_dict(), 'saved_models/predictorhypernet_weights.pth')