#============================================
# Group 1: Assignment 2
# ---------------------
# Tobin Eberle
# Jeff Wheeler
# Ryan Baker
# Tom Wilson
#============================================

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim
import time

from transformers import DistilBertModel, DistilBertTokenizer
import os
import re

#============================================
# Definitions
# -----------
#============================================
BATCH_SIZE = 1
NUM_WORKERS = 1
NUM_EPOCHS = 1
BEST_MODEL_PATH = './best-models/'
TRAIN_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
TEST_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"
VAL_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


#============================================
# Datasets
# --------
# Specifying path and loading the data into dataloaders. Also some vis/system nfo
#============================================

# Read file labels and output text to array
def read_text_files_with_labels(path):
    max_len = 0
    texts = []
    labels = []
    class_folders = sorted(os.listdir(path))  # Assuming class folders are sorted
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            file_names = os.listdir(class_path)
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    file_name_no_ext, _ = os.path.splitext(file_name)
                    text = file_name_no_ext.replace('_', ' ')
                    text_without_digits = re.sub(r'\d+', '', text)
                    if (len(text_without_digits) > max_len):
                        max_len = len(text_without_digits)
                    texts.append(text_without_digits)
                    labels.append(label_map[class_name])

    return np.array(texts), np.array(labels), max_len

# Create text dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

text_train,labels_train, len_train = read_text_files_with_labels(TRAIN_PATH)
text_val,labels_val, len_val = read_text_files_with_labels(VAL_PATH)
text_test,labels_test, len_test = read_text_files_with_labels(TEST_PATH)
max_length = max([len_train, len_val, len_test])

#Datasets
dataset_train = TextDataset(text_train, labels_train, TOKENIZER, max_length)
dataset_val = TextDataset(text_val, labels_val, TOKENIZER, max_length)
dataset_test = TextDataset(text_test, labels_test, TOKENIZER, max_length)

# Data loaders
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

#============================================
# Text Classifier NN
# --------
# Class for creating text classifier based off the in-class examples
#============================================
class TextNN(nn.Module):
    
    def __init__(self, num_classes):
        super(TextNN, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.distilbert(input_ids = input_ids, attention_mask = attention_mask)[0]
        output = self.drop(pooled_output[:, 0])
        return self.out(output)

#============================================
# Training and Validation
# --------
# Training and validation script for NN classes
#============================================

def nnTrainer(model, model_save_name, learning_rate, learning_rate_decay, num_epochs):
    print("Running Training")
    print("#========================================================")
    # Can Edit the script to change your optimizer and scheduler if needed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = learning_rate_decay)
    best_loss = 1e+20

    # Run for set amount of epochs (1 epoch = 1 iteration over entire training data)
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        model.train()
        i = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # Clear gradients accumulated from previous steps
            optimizer.zero_grad()
            # Compute predicitions
            outputs = model(input_ids, attention_mask)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backwards feedback for model parameters wrt loss
            loss.backward()
            # Update model parameters using computer optimizer gradients
            optimizer.step()
            i += 1
            train_loss += loss.item()
        print(f'{epoch + 1},  Train Loss: {train_loss / i:.3f},', end = ' ')

        # Update optimizer learning rate after each epoch
        scheduler.step()

        # Val is similar to train, but we dont need to compute the gradient
        val_loss = 0
        model.eval()
        with torch.no_grad():
            i = 0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                i += 1

            print(f'Val Loss: {val_loss / i:.3f}', end = '\n')

            # Model checkpointing, saves best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(model.state_dict(), BEST_MODEL_PATH + model_save_name)
                best_loss = val_loss

            end_time = time.time()
            print(f'Epoch Calculation Time (s): {(end_time - start_time):.1f}s\n')

    print("Training Completed")
    print("#========================================================")


#============================================
# Testing
# -------
# Calculates the accuracy from our test set
#============================================

def nnTester(model, pth_name):
    print("Running Testing")
    print("#========================================================")
    # Load best saved model
    model.load_state_dict(torch.load((BEST_MODEL_PATH + pth_name), map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Test each image in test_loader
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            # Chose class based off highest probability
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Model Tested: {pth_name}')
    print(f"Total Samples: {total}")
    print(f"Predicted Correct: {correct}")
    print(f'Accuracy of the network on the test images: {(100 * correct / total):.3f} %')

#============================================
# Main
#============================================

print("#========================================================")
print("Script Test Info")
print("#========================================================")

print("Batch Size:\n", BATCH_SIZE)
print("Number of Workers:\n", NUM_WORKERS)
print("Max Token Length:\n", max_length)
print("Train set:\n", len(train_loader)*BATCH_SIZE)
print("\nVal set:\n", len(val_loader)*BATCH_SIZE)
print("\nTest set:\n", len(test_loader)*BATCH_SIZE)
print("\nTrain Loader Size:\n", len(train_loader))
print("\nVal Loader Size:\n", len(val_loader))
print("\nTest Loader Size:\n", len(test_loader))

# Init a device class
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nConnected Device:\n ", device)
print("#========================================================")

# Init a model
distilbert_model = TextNN(4)
# Send our model to the device (GPU or CPU)
distilbert_model.to(device)
nnTrainer(distilbert_model, "distilbert.pth", 0.0001, 0.9, 2)
nnTester(distilbert_model, "distilbert.pth")
print("\n#========================================================")
