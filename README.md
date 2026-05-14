# FGSM & PGD Attacks on Adversarially Trained Models

This repository implements Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks on both baseline and adversarially trained models.

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

## Instruction
```bash
python fgsm_inference.py --model clean
```
- Loads the baseline ResNet18 model with clean weights (`resnet18.pt`)
- Performs FGSM attack with various perturbation rates on one batch of test data
- Saves inference results as PNG images showing original images and adversarial examples at `$\epsilon$=8/255`

```bash
python fgsm_inference.py --model adv
```
- Loads the adversarially trained ResNet18 model (`./fgsm_8_255/trained_fgsm.pt`)
- Saves inference results as PNG images showing original images and adversarial examples at `$\epsilon$=8/255`

```bash
python fgsm_training.py
```
- Loads the baseline ResNet18 model with clean weights (`resnet18.pt`) and trains with adversarial inputs
- Saves the model in folder (`./fgsm_8_255`)

```bash
python pgd_inference.py --model clean
```
- Loads the baseline ResNet18 model with clean weights (`resnet18.pt`)
- Performs PGD attack with various perturbation rates on one batch of test data
- Saves inference results as PNG images showing original images and adversarial examples at `$\epsilon$=8/255` and `$\alpha$=2/255`

```bash
python pgd_inference.py --model adv
```
- Loads the adversarially trained ResNet18 model (`./pgd_8_255/trained_pgd_ep_0.03_alpha_0.007.pt`)
- Saves inference results as PNG images showing original images and adversarial examples at `$\epsilon$=8/255` and `$\alpha$=2/255`

```bash
python pgd_training.py
```
- Loads the baseline ResNet18 model with clean weights (`resnet18.pt`) and trains with adversarial inputs
- Saves the model in folder (`./pgd_8_255`)