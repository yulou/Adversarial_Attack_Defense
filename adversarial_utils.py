import torch

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

def fgsm_attack(model, inputs, labels, epsilon):
    """
    Perform FGSM attack on a batch of inputs.
    
    Args:
        model: Trained model to attack.
        inputs: Original input images (tensor).
        labels: True labels for the inputs (tensor).
        epsilon: Magnitude of the perturbation.

    Returns:
        adversarial_inputs: Adversarial examples generated from inputs.
    """
    # Set model to evaluation mode
    model.eval()

    # Clone inputs and enable gradients
    inputs_adv = inputs.clone().detach().requires_grad_(True)

    # Forward pass: compute predictions and loss
    outputs = model(inputs_adv)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # Backward pass: compute gradient of the loss w.r.t inputs
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    data_grad_sign = inputs_adv.grad.data.sign()

    # Generate adversarial examples by adding perturbation
    adversarial_inputs = inputs_adv + epsilon * data_grad_sign
    
    # Convert to pixel space and clamp
    adv_pixel = adversarial_inputs * std + mean
    # Clip the adversarial examples to be within valid range [0, 1]
    adv_pixel = torch.clamp(adv_pixel, 0, 1)
    
    # Renormalize
    adv_final = (adv_pixel - mean) / std

    # Switch back to training mode
    model.train()

    return adv_final.detach()

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

def evaluate_fgsm(model, test_loader, device, epsilon):
    """
    Evaluate model on test dataset.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test dataset
        device: Device to run evaluation on
        epsilon: Attack perturbation rate
    
    Returns:
        clean_acc: The prediction accuracy on clean inputs
        clean_loss_avg: Average loss on clean inputs
        adv_acc: The prediction accuracy on adversarial inputs
        adv_loss_avg: Average loss on adversarial inputs
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
            # Note: fgsm_attack requires gradient computation, so we need to handle it carefully
            # We can't use torch.no_grad() for the attack generation
            pass

    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # 2. Adversarial evaluation
        adv_inputs = fgsm_attack(model, inputs, labels, epsilon)
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

    return clean_acc, clean_loss_avg, adv_acc, adv_loss_avg

def evaluate_pgd(model, test_loader, device, epsilon, alpha, num_iter):
    """
    Evaluate model on test dataset.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test dataset
        device: Device to run evaluation on
        epsilon: Attack perturbation rate
        alpha: Step size for each iteration.
        num_iter: Number of iterations for the attack.
    
    Returns:
        clean_acc: The prediction accuracy on clean inputs
        clean_loss_avg: Average loss on clean inputs
        adv_acc: The prediction accuracy on adversarial inputs
        adv_loss_avg: Average loss on adversarial inputs
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
    
    return clean_acc, clean_loss_avg, adv_acc, adv_loss_avg