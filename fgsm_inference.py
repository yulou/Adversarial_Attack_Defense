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

#model.load_state_dict(torch.load('resnet18.pt'))
#model.load_state_dict(torch.load('trained_fgsm.pt'))

model.to(device)

# Transform the test dataset so that the inputs align with the training sample distribution
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
batch_size = 8
# Create loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model.eval()

accuracy = test_natural(model, test_loader)
print(f"Natural Test Accuracy: {accuracy:.2f}%")


EP = 8/255

clean_acc, clean_loss_avg, adv_acc, adv_loss_avg = evaluate_fgsm(model, test_loader, device, EP)
print(f"Test Clean Acc: {clean_acc:.2f}%, Test Adv Acc: {adv_acc:.2f}%")

initial_accuracy = accuracy

#epsilon_arr = [4/255, 8/255, 16/255, 38/255]
#
#adv_acc_arr = []
#for ep in epsilon_arr:
#    clean_acc, clean_loss_avg, adv_acc, adv_loss_avg = evaluate(model, test_loader, device, ep)
#    adv_acc_arr.append(adv_acc)

#print("epsilon_arr: {}".format(epsilon_arr))
#print("adv_acc_arr: {}".format(adv_acc_arr))
#import pickle
#with open('FGSM_training_testing_summary.pkl', 'wb') as f:
#    pickle.dump(summary, f)

def plot_attack_classification(model, data_loader, epsilon, class_names):
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

    # Generate adversarial examples
    inputs = inputs.to(device)
    labels = labels.to(device)
    adversarial_inputs = fgsm_attack(model, inputs, labels, epsilon)

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
    fig, axes = plt.subplots(3, batch_size, figsize=(12, 6))
    
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
        axes[2, i].imshow((adversarial_inputs-inputs)[i].transpose(1, 2, 0) * 0.5 + 0.5)
        axes[2, i].axis('off')
    
    fig.suptitle(f"FGSM Attack Results with Epsilon = {epsilon}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'FGSM_result_eps_{epsilon:.2f}.png')
    plt.show()

# Class names for CIFAR-10
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

plot_attack_classification(model, test_loader, epsilon=EP, class_names=class_names)