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

# Get the absolute path to PyTorch_CIFAR10 folder
pytorch_cifar10_path = '/home/yulou/Homework/ARIN5303_AI_Cybersecurity/Project/PyTorch_CIFAR10/'

# Add to Python path (BEFORE importing)
#sys.path.append(pytorch_cifar10_path)

from resnet import resnet18

# Check device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

#model = torch.hub.load("../pytorch-cifar-models", "cifar10_resnet20", source='local', pretrained=True)
#weights = torch.load("cifar10_resnet20-4118986f.pt")
#model.load_state_dict(weights)

model = resnet18(pretrained=True)
#model.load_state_dict(torch.load('best_finetuned_pgd_ep_0.03_alpha_0.01.pt'))

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

# Define transforms
#train_transform = transforms.Compose([
#    transforms.Resize((32, 32)),
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])

# Use ImageFolder for your directory structure
#train_dataset = torchvision.datasets.ImageFolder(
#    root='./data/train',  # Your train folder with class subdirectories
#    transform=train_transform
#)
#print(f"train_dataset len {len(train_dataset)}")
#test_dataset = torchvision.datasets.ImageFolder(
#    root='./data/test',   # Your test folder with class subdirectories
#    transform=train_transform
#)
print(f"train_dataset len {len(train_dataset)}")
print(f"test_dataset len {len(test_dataset)}")

batch_size=8
# Create loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model.eval()

def test_natural(net, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_natural(net, train_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

accuracy = train_natural(model, train_loader)
print(f"Natural Train Accuracy: {accuracy:.2f}%")

accuracy = test_natural(model, test_loader)
print(f"Natural Test Accuracy: {accuracy:.2f}%")

def evaluate(model, test_loader, device, epsilon, alpha, num_iter):
    """
    Evaluate model on test dataset.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test dataset
        device: Device to run evaluation on
        return_details: If True, returns detailed metrics dictionary
    
    Returns:
        If return_details=False: Tuple of (accuracy, loss)
        If return_details=True: Dictionary with accuracy, loss, and other metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    clean_loss = 0.0
    adv_loss = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            # 1. Clean evaluation
            outputs_clean = model(inputs)
            
            # Calculate loss
            loss_clean = criterion(outputs_clean, labels)
            clean_loss += loss_clean.item()
            
            # Calculate accuracy
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            correct_clean += (predicted_clean == labels).sum().item()
            # 2. Adversarial evaluation
            # Note: pgd_attack requires gradient computation, so we need to handle it carefully
            # We can't use torch.no_grad() for the attack generation
            pass

    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # 2. Adversarial evaluation
        adv_inputs = pgd_attack(model, inputs, labels, epsilon, alpha, num_iter)
        # make sure model in eval() mode
        model.eval()

        outputs_adv = model(adv_inputs)
        loss_adv = criterion(outputs_adv, labels)
        adv_loss += loss_adv.item()

        _, predicted_adv = torch.max(outputs_adv, 1)
        correct_adv += (predicted_adv == labels).sum().item()
        total += labels.size(0)

    
    # Calculate metrics
    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    clean_loss_avg = clean_loss / len(test_loader)
    adv_loss_avg = adv_loss / len(test_loader)
    print(f"Test Clean Acc: {clean_acc:.2f}%, Test Adv Acc: {adv_acc:.2f}%")
    return clean_acc, clean_loss_avg, adv_acc, adv_loss_avg

def pgd_attack(model, inputs, labels, epsilon, alpha, num_iter):
    """
    Perform PGD attack on a batch of inputs.
    
    Args:
        model: Trained model to attack.
        inputs: Original input images (tensor).
        labels: True labels for the inputs (tensor).
        epsilon: Maximum perturbation.
        alpha: Step size for each iteration.
        num_iter: Number of iterations for the attack.

    Returns:
        adversarial_inputs: Adversarial examples generated from inputs.
    """
    # Set model to evaluation mode
    model.eval()

    # Clone inputs and enable gradients
    inputs_adv = inputs.clone().detach()

    for _ in range(num_iter):
        inputs_adv.requires_grad = True
        # Forward pass: compute predictions and loss
        outputs = model(inputs_adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass: compute gradient of the loss w.r.t inputs
        model.zero_grad()
        loss.backward()

        # Get gradient sign
        data_grad_sign = inputs_adv.grad.sign()

        with torch.no_grad():
            # Generate adversarial examples by adding perturbation
            inputs_adv = inputs_adv + alpha * data_grad_sign

            # Project back to epsilon ball around original inputs (in normalized space)
            eta = torch.clamp(inputs_adv - inputs, -epsilon, epsilon)
            inputs_adv = inputs + eta
    
            # Convert to pixel space and clamp
            adv_pixel = inputs_adv * std + mean
            adv_pixel = torch.clamp(adv_pixel, 0, 1)
            
            # Renormalize
            inputs_adv = (adv_pixel - mean) / std
            inputs_adv = inputs_adv.detach()

    # Switch back to training mode
    model.train()

    return inputs_adv

num_epochs = 20
EP = 8/255

alpha = 2/255 
num_iter = 7

clean_acc, clean_loss_avg, adv_acc, adv_loss_avg = evaluate(model, test_loader, device, EP, alpha, num_iter)

initial_accuracy = accuracy

#epsilon_arr = [4/255, 8/255, 16/255, 38/255]
#adv_acc_arr = []
#for ep in epsilon_arr:
#    clean_acc, clean_loss_avg, adv_acc, adv_loss_avg = evaluate(model, test_loader, device, ep, alpha, num_iter)
#    adv_acc_arr.append(adv_acc)
#
#print("epsilon_arr: {}".format(epsilon_arr))
#print("adv_acc_arr: {}".format(adv_acc_arr))

#import pickle
#with open('PGD_training_testing_summary.pkl', 'wb') as f:
#    pickle.dump(summary, f)

# Save the adversarially trained model
#torch.save(model.state_dict(), f"trained_pgd_ep_{epsilon:.2f}_alpha_{alpha}.pt")
#print("pgd trained model saved.")

def plot_attack_classification1(model, data_loader, epsilon, alpha, num_iter, class_names):
    """
    Visualize the impact of FGSM attacks on classification.
    
    Args:
        model: Trained model to attack.
        data_loader: DataLoader for test data.
        epsilon: Magnitude of the FGSM perturbation.
        class_names: List of class names corresponding to CIFAR-10 dataset.
    """
    data_iter = iter(data_loader)
    inputs, labels = next(data_iter)
    inputs, labels = next(data_iter)
    inputs, labels = next(data_iter)
    # Generate adversarial examples
    inputs = inputs.to(device)
    labels = labels.to(device)
    adversarial_inputs = pgd_attack(model, inputs, labels, epsilon, alpha, num_iter)

    # Get predictions for both original and adversarial examples
    model.eval()
    with torch.no_grad():
        outputs_original = model(inputs)
        outputs_adversarial = model(adversarial_inputs)
        _, predicted_original = torch.max(outputs_original.data, 1)
        _, predicted_adversarial = torch.max(outputs_adversarial.data, 1)

    inputs = inputs * std + mean
    adversarial_inputs = adversarial_inputs * std + mean
    # Convert tensors to numpy for visualization
    inputs = inputs.cpu().detach().numpy()
    adversarial_inputs = adversarial_inputs.cpu().detach().numpy()

    # Plot original and adversarial images with labels
    fig, axes = plt.subplots(2, batch_size, figsize=(12, 6))
    
    for i in range(batch_size):
        # Original images
        axes[0, i].imshow(inputs[i].transpose(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].set_title(f"Real: {class_names[labels[i].item()]}\nPred: {class_names[predicted_original[i].item()]}")
        axes[0, i].axis('off')

        # Check if adversarial prediction is correct
        is_correct = (predicted_adversarial[i] == labels[i])
        
        # Set title color based on prediction correctness
        if is_correct:
            title_color = 'green'  # Correct prediction (still robust)
            status = "✓ CORRECT"
        else:
            title_color = 'red'    # Misclassified
            status = "✗ MISCLASSIFIED"
        
        # Adversarial images with colored title
        axes[1, i].imshow(adversarial_inputs[i].transpose(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].set_title(f"Under Attack\nPred: {class_names[predicted_adversarial[i].item()]}\n{status}", 
                            color=title_color, fontweight='bold')
        axes[1, i].axis('off')
    
    fig.suptitle(f"PGD Attack Results with Epsilon = {epsilon}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'PGD_result_eps_{epsilon:.2f}.png')
    plt.show()

# Class names for CIFAR-10
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

#plot_attack_classification(model, test_loader, epsilon=EP, class_names=class_names)
plot_attack_classification1(model, test_loader, epsilon=EP, alpha=alpha, num_iter=num_iter, class_names=class_names)