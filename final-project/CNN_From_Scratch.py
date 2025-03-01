import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import torchvision
from torchvision.models import resnet18  # pretrained model
from torchvision import models, transforms, datasets

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from save_datasets import KaggleDataset


# ============================================
# TransferLearning Network
# ============================================
class CNN(nn.Module):
    def __init__(self, imageModel, num_classes):
        super(CNN, self).__init__()
        # Image path
        self.imageModel = imageModel
        # Setup for transfer learning
        # self.imageModel.eval()
        for param in self.imageModel.parameters():
            # Turn on weight updating
            param.requires_grad = True
        self.imageModel.fc = nn.Linear(512, num_classes)

    def forward(self, image):
        # Image path
        return self.imageModel(image)


# ============================================
# Training/Validtion/Testing Functions
# ============================================
def train(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in iterator:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(iterator)


# Define evaluation function
def evaluate(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in iterator:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item()

    return total_loss / len(iterator)


def predict(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    predictions = []
    classes = []

    with torch.no_grad():  # Disable gradient tracking
        for batch in dataloader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            predictions.extend(preds.cpu().numpy())
            classes.extend(labels.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return predictions, classes


if __name__ == "__main__":
    TRAIN_PATH = "/work/TALC/enel645_2025w/group1-dataset/kaggle_real_ai_dataset/train.pt"
    TEST_PATH = "/work/TALC/enel645_2025w/group1-dataset/kaggle_real_ai_dataset/test.pt"
    VAL_PATH = (
        "/work/TALC/enel645_2025w/group1-dataset/kaggle_real_ai_dataset/val.pt"
    )
    BEST_MODEL_PATH = "./best-models/"
    NUM_WORKERS = 2  # Max 2 cpus

    # ============================================
    # Tuneable Parameters
    # ============================================
    NUM_EPOCHS = 6
    BATCH_SIZE = 24
    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    # DROPOUT = 0.3
    # Best model save name
    SAVE_NAME = "tl_cnn_best_model.pth"

    print("#========================================================")
    print("NUM_EPOCHS = ", NUM_EPOCHS)
    print("BATCH_SIZE = ", BATCH_SIZE)
    print("WEIGHT_DECAY = ", WEIGHT_DECAY)
    print("LEARNING_RATE = ", LEARNING_RATE)
    print("SAVE_NAME = ", SAVE_NAME)
    # print("DROPOUT = ", DROPOUT)
    print("#========================================================")

    # Create dataloaders
    try:
        print("Loading datasets")
        train_dataset = torch.load(TRAIN_PATH, weights_only=False)
        val_dataset = torch.load(VAL_PATH, weights_only=False)
        test_dataset = torch.load(TEST_PATH, weights_only=False)
    except Exception as e:
        print("Failed to load datasets")
        print(e)
        exit()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    k_batch = next(iter(train_loader))
    for i in range(3):
        img = k_batch["image"].numpy()[i].transpose(1, 2, 0)
        label = k_batch["label"].numpy()[i]
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    fig.text(
        0.04,
        0.75,
        "Kaggle Images",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
    )
    plt.savefig("test-images.png")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Connected device: ", device)
    print("\n")
    imageModel = resnet18(weights=None)
    model = CNN(imageModel, 2).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_loss = 1e10  # best loss tracker
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.5f}")
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch: {epoch+1}, Val Loss: {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH + SAVE_NAME)
            print("Saving Model")
        end_time = time.time()
        print(f"Epoch Calculation Time (s): {(end_time - start_time):.1f}s\n")
    model.load_state_dict(torch.load(BEST_MODEL_PATH + SAVE_NAME))

    # Evaluation
    test_predictions, test_classes = predict(model, test_loader, device)
    equal_count = 0
    for pred, class_ in zip(test_predictions, test_classes):
        if pred == class_:
            equal_count += 1

    print("Correct Predictions: ", equal_count)
    print("Total Labels: ", len(test_classes))
