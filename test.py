import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hyperparams
BATCH_SIZE = 512
LR = 0.005
NUM_WORKERS = 0
EPOCHS=50
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.device_count() > 0:
        print("USE", torch.cuda.device_count(), "GPUs!")
    else:
        print("USE ONLY CPU!")
DEVICE = get_default_device()

normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(normMean, normStd)

trainTransform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normTransform
    ])
testTransform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        normTransform
    ])

train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=trainTransform)
test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=testTransform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

model = torchvision.models.resnet18(pretrained=True)
num_fltrs = model.fc.in_features
model.fc = nn.Sequential(
            nn.Linear(num_fltrs , 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Hardswish(),
            nn.Linear(512 , 10),
            nn.Softmax(dim=1))
model = model.to(DEVICE)
print(model)
criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), LR,
#                             momentum=MOMENTUM,
#                             weight_decay=WEIGHT_DECAY, nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY, amsgrad=False)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.005, cycle_momentum=False)

def trainEpoch(device, model, loader, optimizer, criterion):
    grad_clip = 0.1
    loop = tqdm(loader)
    model.train()

    losses = []
    correct = 0
    total = 0
    for batch_id, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        model = model.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        loop.set_postfix(train_loss=loss.item(), train_accuracy=(100 * correct / total))

    return np.mean(losses), 100 * correct / total


def testEpoch(device, model, loader, criterion):
    model.eval()
    with torch.no_grad():
        losses = []
        correct = 0
        total = 0
        loop = tqdm(loader)
        for batch_id, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            model = model.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loop.set_postfix(test_loss=loss.item(), test_accuracy=(100 * correct / total))

        return np.mean(losses), 100 * correct / total


last_best = 0
train_losses = []
train_acc = []
test_losses = []
test_acc = []
for epoch in range(EPOCHS):

    train_loss, train_accuracy = trainEpoch(DEVICE, model, train_loader, optimizer, criterion)
    test_loss, test_accuracy = testEpoch(DEVICE, model, test_loader, criterion)

    train_losses.append(train_loss)
    train_acc.append(train_accuracy)
    test_losses.append(test_loss)
    test_acc.append(test_accuracy)

    scheduler.step()

    if last_best <= test_accuracy:
        torch.save(model.state_dict(), 'resnet_50_pretrained_baseline.pt')
        print("Saving new Best Model!")
        last_best = test_accuracy

    print("Train losses: {0:.3f},   Train acc: {1:.3f} ".format(train_losses[-1], train_acc[-1]))
    print("Test losses: {0:.3f},   Test acc: {1:.3f} ".format(test_losses[-1], test_acc[-1]))

dict = {'train_loss': train_losses, 'train_accuracy': train_acc, 'test_loss': test_losses, 'test_accuracy': test_acc}
df = pd.DataFrame(dict)
df.to_csv('loss_and_acc_resnet18_baseline.csv')