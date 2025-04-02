import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import torchvision
from torchvision import models, transforms, datasets

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from save_datasets import KaggleDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


#============================================
# Fully Connected Neural Network
#============================================
class FCNN(nn.Module):
    def __init__(self, imageDimensions, num_classes):
        super(FCNN, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(imageDimensions, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Dropout layers - left out of model since they decreased accuracy
        # self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        input = input.view(input.size(0), -1) # flatten input
        input = F.relu(self.bn1(self.fc1(input)))
        # input = self.dropout(input)
        input = F.relu(self.bn2(self.fc2(input)))
        # input = self.dropout(input)
        input = F.relu(self.bn3(self.fc3(input)))
        input = self.fc4(input)
        return input


#============================================
# Training/Validtion/Testing Functions
#============================================
def train(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in iterator:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

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
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

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

            images = batch['image'].to(device)
            labels = batch['label'].to(device)


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
    VAL_PATH = "/work/TALC/enel645_2025w/group1-dataset/kaggle_real_ai_dataset/val.pt"
    BEST_MODEL_PATH = './best-models/'
    NUM_WORKERS = 2 # Max input 2 cpus

    #============================================
    # Tuneable Parameters
    #============================================
    NUM_EPOCHS = 10
    BATCH_SIZE = 24
    WEIGHT_DECAY = 0.00001
    LEARNING_RATE = 0.0001
    SAVE_NAME = "fcnn_best_model.pth"

    print("#========================================================")
    print("NUM_EPOCHS = ", NUM_EPOCHS)
    print("BATCH_SIZE = ", BATCH_SIZE)
    print("WEIGHT_DECAY = ", WEIGHT_DECAY)
    print("LEARNING_RATE = ", LEARNING_RATE)
    print("SAVE_NAME = ", SAVE_NAME)
    print("#========================================================")

    # Create dataloaders
    try:
        print("Loading datasets")
        train_dataset = torch.load(TRAIN_PATH, weights_only = False)
        val_dataset = torch.load(VAL_PATH, weights_only = False)
        test_dataset = torch.load(TEST_PATH,weights_only = False)
    except Exception as e:
        print("Failed to load datasets")
        print(e)
        exit()

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Connected device: ", device)
    print("\n")

    # Get the image size
    train_sample = DataLoader(train_dataset, batch_size = 1, shuffle = False, num_workers = NUM_WORKERS)
    batch = next(iter(train_sample))
    image_key = ''
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            image_key = key
            break

    if image_key is None:
        raise ValueError("No image data found in the batch.")

    image = batch[image_key]
    _, channels, height, width = image.shape

    # Model
    imageDimensions = channels * height * width
    model = FCNN(imageDimensions, 2).to(device)  # 2 classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion = nn.CrossEntropyLoss() 

    # Training loop
    best_loss = 1e+10 # best loss tracker
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.5f}')
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch + 1}, Val Loss: {val_loss:.5f}')
        
        scheduler.step(val_loss)  # This adjusts the learning rate based on val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH + SAVE_NAME)
            print("Saving Model")
        end_time = time.time()
        print(f'Epoch Calculation Time (s): {(end_time - start_time):.1f}s\n')
    model.load_state_dict(torch.load(BEST_MODEL_PATH + SAVE_NAME))

    # Evaluation
    test_predictions, test_classes = predict(model, test_loader, device)
    equal_count = 0
    for pred, class_ in zip(test_predictions, test_classes):
        if pred == class_:
            equal_count += 1
            
    print("Correct Predictions: ", equal_count)
    print("Total Labels: ", len(test_classes) )