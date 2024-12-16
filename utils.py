import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, path_to_model_save):
    """
    Trains the model and evaluates on the validation set.

    Parameters:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (loss): Loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on (CPU/GPU).

    Returns:
        nn.Module: Trained model.
    """
    model.to(device)
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), path_to_model_save)
            print('\n Model saved')

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

    return model

def evaluate_model(model, test_loader, device, num_classes, path_to_best_model):
    """
    Evaluates the model on the test set, computes accuracy, and calculates AUPRC.

    Parameters:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on (CPU/GPU).
        num_classes (int): Number of classes.

    Returns:
        None
    """
    model.load_state_dict(torch.load(path_to_best_model))
    model.eval()
    model.to(device)

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get model outputs and probabilities
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            preds = torch.argmax(probs, dim=1)  # Predicted class indices

            # Collect all labels, probabilities, and predictions
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Combine all batches
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate AUPRC for each class
    average_precisions = []
    for i in range(num_classes):
        # Binary labels for the current class
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]

        # Precision-Recall calculation
        precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
        ap = average_precision_score(binary_labels, class_probs)
        average_precisions.append(ap)

        # Plot Precision-Recall curve
        plt.plot(recall, precision, label=f"Class {i} (AP = {ap:.2f})")

    # Plot formatting
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.savefig('/home/sameer/Cough-Sense/figures/results1.png')
    plt.show()

    # Print AUPRC scores
    for i, ap in enumerate(average_precisions):
        print(f"AUPRC for Class {i}: {ap:.4f}")
    print(f"Mean AUPRC: {np.mean(average_precisions):.4f}")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=['neither', 'viral', 'bacterial'])
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix without seaborn
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    plt.savefig('/home/sameer/Cough-Sense/figures/confusion_matrix.png')
    plt.show()