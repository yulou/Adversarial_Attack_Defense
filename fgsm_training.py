import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import argparse

from resnet import resnet18

from adversarial_utils import fgsm_attack, evaluate_fgsm, test_natural

# Check device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

model = resnet18(pretrained=True)

model.to(device)

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    return torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    return torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
std = torch.Tensor(cifar10_std).view(1,3,1,1).to(device)
mean = torch.Tensor(cifar10_mean).view(1,3,1,1).to(device)
# Define data transformations
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)

img, label = test_dataset[0]
print("img shape {}, label {}, range min {}, max {}".format(img.shape, label, torch.min(img), torch.max(img)))

print(f"train_dataset len {len(train_dataset)}")
print(f"test_dataset len {len(test_dataset)}")

# Create loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

model.eval()

accuracy = test_natural(model, train_loader)
print(f"Natural Train Accuracy: {accuracy:.2f}%")

accuracy = test_natural(model, test_loader)
print(f"Natural Test Accuracy: {accuracy:.2f}%")


def adversarial_training(
    model, train_loader, test_loader, num_epochs, epsilon
):
    """
    Train the model using adversarial training with PGD.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training dataset.
        optimizer: Optimizer for training.
        num_epochs: Number of training epochs.
        epsilon: Maximum perturbation for PGD attack.

    Returns:
        Trained model.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=2)
    best_test_acc = 0
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    train_history = {'loss': [], 'clean_acc': [], 'adv_acc': []}
    test_history = {'clean_loss': [], 'adv_loss': [], 'clean_acc': [], 'adv_acc': []}
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Generate adversarial examples using PGD
            adversarial_inputs = fgsm_attack(model, inputs, labels, epsilon)

            # Combine clean and adversarial examples
            combined_inputs = torch.cat([inputs, adversarial_inputs], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(combined_inputs)
            loss = criterion(outputs, combined_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracies for monitoring
            with torch.no_grad():
                # Clean accuracy
                clean_outputs = model(inputs)
                _, clean_predicted = torch.max(clean_outputs.data, 1)
                correct_clean += (clean_predicted == labels).sum().item()
                
                # Adversarial accuracy
                adv_outputs = model(adversarial_inputs)
                _, adv_predicted = torch.max(adv_outputs.data, 1)
                correct_adv += (adv_predicted == labels).sum().item()
                
                total += labels.size(0)
            running_loss += loss.item()

        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        clean_acc = 100 * correct_clean / total
        adv_acc = 100 * correct_adv / total
        
        train_history['loss'].append(epoch_loss)
        train_history['clean_acc'].append(clean_acc)
        train_history['adv_acc'].append(adv_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Clean Accuracy: {clean_acc:.2f}%")
        print(f"  Adversarial Accuracy: {adv_acc:.2f}%\n")

        clean_acc, clean_loss_avg, adv_acc, adv_loss_avg = evaluate_fgsm(model, test_loader, device, epsilon)
        test_history['clean_loss'].append(clean_loss_avg)
        test_history['adv_loss'].append(adv_loss_avg)
        test_history['clean_acc'].append(clean_acc)
        test_history['adv_acc'].append(adv_acc)
        print(f"Epoch {epoch}: Test Clean Acc: {clean_acc:.2f}%, Test Adv Acc: {adv_acc:.2f}%")
        
        # test accuracy is the average of clean and adversarial cases
        test_acc = (clean_acc + adv_acc) / 2
        test_loss = (clean_loss_avg + adv_loss_avg) / 2
        # Early stopping if test accuracy drops too much
        #if test_acc < best_test_acc - 1:  # Drop more than 1%
        #    print(f"Test accuracy dropped from {best_test_acc:.2f}% to {test_acc:.2f}%. Stopping!")
        #    break
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'./fgsm_8_255/best_finetuned_fgsm_ep{epsilon:.2f}.pt')
        
        scheduler.step(test_loss)

    return model, train_history, test_history

num_epochs = 10
epsilon = 8/255  # Maximum perturbation
# 0.0157	4/255  Smaller Perturbation. 
#A good starting point for a "lighter" defense if the standard 8/255 causes
# too much of a drop in accuracy on clean images

# 0.0314    8/255  The Standard Benchmark. This is the most common value in 
# adversarial robustness literature and is the primary value you should train 
# and evaluate with. It is large enough to be a strong challenge but small enough 
# to be imperceptible.

# 0.0627	16/255 Larger Perturbation. This will create much stronger and more 
# obvious adversarial examples. Training with this value will result in a model 
# that is robust to very large changes but at the cost of a much larger drop in 
# accuracy on clean, unperturbed images.

# 0.13 – 0.15	~33-38/255 Extreme Challenge. This is used in some experiments 
# to really stress-test a model's defenses. At this level, the model's performance 
# on clean images is likely to suffer significantly, and it is typically used for 
# research purposes rather than practical applications
# Train the model
print("Starting Training...")
model, train_history, test_history = adversarial_training(model, train_loader, test_loader, num_epochs, epsilon)

summary = {}
summary['train_history'] = train_history
summary['test_history'] = test_history
summary['epsilon'] = epsilon
summary['attack'] = "FGSM"

import pickle
with open('FGSM_training_testing_summary.pkl', 'wb') as f:
    pickle.dump(summary, f)

# Save the adversarially trained model
torch.save(model.state_dict(), f"./fgsm_8_255/trained_fgsm_ep_{epsilon:.2f}.pt")
print("fgsm trained model saved.")