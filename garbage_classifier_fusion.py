#============================================
# Group 1: Assignment 2
# ---------------------
# Tobin Eberle
# Jeff Wheeler
# Ryan Baker
# Tom Wilson
#============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


import torchvision
from torchvision.models import resnet18 # pretrained model
from torchvision import models, transforms, datasets

from transformers import DistilBertModel, DistilBertTokenizer

import os
import re
import numpy as np
import time

TRAIN_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
TEST_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"
VAL_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
BEST_MODEL_PATH = './best-models/'
NUM_WORKERS = 2 # Max 2 cpus
MAX_LENGTH = 200

#============================================
# Tuneable Parameters
#============================================
NUM_EPOCHS = 3
BATCH_SIZE = 16
NUM_FUSION_FEATURES = 100
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.01
DROPOUT = 0.4
#Best model save name
SAVE_NAME = "best_fusion_model.pth"

#============================================
# Datasets
#============================================
# Need to return image, tokenized text, and label
class FusionDataSet(Dataset):
    def __init__(self, transform, tokenizer, dir, max_len):
        self.transform = transform
        self.tokenizer = tokenizer
        self.dir = dir
        self.max_length = max_len
        self.images = np.empty(0)
        self.labels = np.empty(0)
        self.texts = np.empty(0)
        
        # Based off in-class example to extract filenames
        def filename_to_text(file):
            file_no_ext, _ = os.path.splitext(file)
            text = file_no_ext.replace('_', ' ')
            text_without_digits = re.sub(r'\d+', '', text)
            return text_without_digits
        
        # Also based off inclass example to sort through the file structure
        def get_data():
            classes = sorted(os.listdir(self.dir))
            label_map = {class_name: idx for idx, class_name in enumerate(classes)}

            for class_name in classes:
                class_path = os.path.join(self.dir, class_name)
                if os.path.isdir(class_path):
                    file_names = os.listdir(class_path)
                    for file_name in file_names:
                        img_path = os.path.join(class_path, file_name)
                        if os.path.isfile(img_path):
                            self.images = np.append(self.images, img_path)
                            self.texts = np.append(self.texts, filename_to_text(img_path))
                            self.labels = np.append(self.labels, label_map[class_name])
        get_data()

    def __len__(self):
        return len(self.labels)

    # Returns a sample from dataset based on given idx (for dataloader)
    def __getitem__(self, idx):
        image = self.images[idx]
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Need to return the image as tensor
        image = torchvision.io.decode_image(image, mode= "RGB")
        image = transforms.functional.to_pil_image(image)
        image = self.transform(image)
       
        # Need to return text as tensor
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create datasets
train_dataset = FusionDataSet(transform, tokenizer, TRAIN_PATH, MAX_LENGTH)
val_dataset = FusionDataSet(transform, tokenizer, VAL_PATH, MAX_LENGTH)
test_dataset = FusionDataSet(transform, tokenizer, TEST_PATH, MAX_LENGTH)

# print("Train Dataset: ", len(train_dataset))
# print("Val Dataset: ", len(val_dataset))
# print("Test Dataset: ", len(test_dataset))

# Create loaders
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

#============================================
# Fusion Network
#============================================

class FusionNetwork(nn.Module):
    def __init__(self, imageModel, textModel, num_classes, num_fusion_features):
        super(FusionNetwork, self).__init__()
        # Image path
        self.imageModel = imageModel
        # Setup for transfer learning
        self.imageModel.eval()
        for param in self.imageModel.parameters():
            # Turn off weight updating
            param.requires_grad = False

        self.imageModel.fc = nn.Linear(512, num_fusion_features)
        self.fc_image_norm = nn.LayerNorm(num_fusion_features)
        
        # Text path
        self.textModel = textModel
        self.fc_text = nn.Linear(self.textModel.config.hidden_size, num_fusion_features)
        self.fc_text_norm = nn.LayerNorm(num_fusion_features)
        
        # Dense fusion layer & Dropout
        self.drop = nn.Dropout(DROPOUT)
        self.fusion_classifier = nn.Linear(num_fusion_features * 2, num_classes)


    def forward(self, image, text, attention_mask):
        # Image path
        image = self.imageModel(image)
        image_norm = self.fc_image_norm(image)

        # Text path
        text = self.textModel(input_ids = text, attention_mask = attention_mask)
        text = text[0]
        text = self.fc_text(text[:, 0])
        text_norm = self.fc_text_norm(text)

        # Fusion path (concatenate horizontally)
        x = torch.cat((image_norm, text_norm), dim = 1)
        # Do we need to flatten?
        x = self.drop(x)
        return self.fusion_classifier(x)

#============================================
# Training/Validtion/Testing Functions
#============================================
# As per the online text example

def train(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in iterator:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(images, input_ids, attention_mask)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(images, input_ids, attention_mask)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            labels = batch['label'].to(device)


            # Forward pass
            outputs = model(images, input_ids, attention_mask)

            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels).item()  
            total += labels.size(0)  
            predictions.extend(preds.cpu().numpy())
            classes.extend(labels.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return predictions, classes

#============================================
# Main
#============================================
# Print our tuneable parameters

print("#========================================================")
print("NUM_EPOCHS = ", NUM_EPOCHS)
print("BATCH_SIZE = ", BATCH_SIZE)
print("NUM_FUSION_FEATURES = ", NUM_FUSION_FEATURES)
print("WEIGHT_DECAY = ", WEIGHT_DECAY)
print("LEARNING_RATE = ", LEARNING_RATE)
print("DROPOUT = ", DROPOUT)
print("#========================================================")

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Connected device: ", device)
print("\n")
imageModel = resnet18(weights='ResNet18_Weights.DEFAULT')
textModel = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = FusionNetwork(imageModel, textModel, 4, NUM_FUSION_FEATURES).to(device)

# Training parameters
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# Code Test loop
# for epoch in range(NUM_EPOCHS):
#     start_time = time.time()
#     train_loss = train(model, val_loader, optimizer, criterion, device)
#     print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
#     val_loss = evaluate(model, test_loader, criterion, device)
#     print(f'Epoch: {epoch+1}, Val Loss: {val_loss:.4f}')
#     if val_loss < best_loss:
#         best_loss = val_loss
#         torch.save(model.state_dict(), BEST_MODEL_PATH + 'best_model.pth')
#         print("Saving Model")
#     end_time = time.time()
#     print(f'Epoch Calculation Time (s): {(end_time - start_time):.1f}s\n')

# def data(iterator):

#     for batch in iterator:
#         images = batch['image'].to(device)
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)

#         print(images)
#         print(input_ids)
#         print(labels)
# data(train_loader)

# Training loop
best_loss = 1e+10 # best loss tracker
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, device)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.5f}')
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f'Epoch: {epoch+1}, Val Loss: {val_loss:.5f}')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH + SAVE_NAME)
        print("Saving Model")
    end_time = time.time()
    print(f'Epoch Calculation Time (s): {(end_time - start_time):.1f}s\n')

model.load_state_dict(torch.load(BEST_MODEL_PATH + SAVE_NAME))
# Evaluation
test_predictions, test_classes = predict(model, test_loader, device)
print("Correct Predictions: ", np.sum(test_predictions == test_classes))
print("Total Labels: ", len(test_classes) )
