#============================================
# Group 1: Assignment 2
# ---------------------
# Tobin Eberle
# Jeff Wheeler
# Ryan Baker
# Tom Wilson
#============================================

import torch
from torchvision.models import resnet18 # pretrained model
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim
import time

#============================================
# Definitions
# -----------
#============================================
BATCH_SIZE = 16 
NUM_WORKERS = 1
NUM_EPOCHS = 5
BEST_MODEL_PATH = './best-models/'
TRAIN_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
TEST_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"
VAL_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"

#============================================
# Transforms
# ----------
# The required pre transforms for images
#============================================

#Think about using transforms.v2 for this (as per pytorch website)
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#============================================
# Datasets
# --------
# Specifying path and loading the data into dataloaders. Also some vis/system nfo
#============================================

# Image loaders (loads the image folders into datasets with the folder name as their class)
train_dataset = datasets.ImageFolder(root = TRAIN_PATH, transform = data_transforms)
test_dataset = datasets.ImageFolder(root = TEST_PATH, transform = data_transforms)
val_dataset = datasets.ImageFolder(root = VAL_PATH, transform = data_transforms)

# Data loaders, take the imageset and returns batches of images and corresponding labels
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

#Save an example image from the dataset
def image_show(training_batch, img_num, class_names):
    img = training_batch[0].numpy()[img_num].transpose(1,2,0)
    img_class = train_batch[1].numpy()[img_num]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    plt.figure()
    plt.imshow((img * 255).astype(np.uint8))
    plt.title(class_names[img_class])
    plt.savefig("example_batch_image.png")

#============================================
# RESNET CNN
# --------
# Class for creating a CNN, based off the in-class examples
#============================================

class resnetCNN(nn.Module):
    def __init__(self, num_classes, input_shape, transfer = False):
        super().__init__()

        # Always four for this classifier
        self.num_classes = num_classes
        # (channels, height, width)
        self.input_shape = input_shape
        # Do we transfer pre-trained weights
        self.transfer = transfer
        # Load pretrained weights if transfer = true, otherwise init weights randomly
        self.feature_extractor = resnet18(weights='ResNet18_Weights.DEFAULT')
        # Get the number of output features from the convolution layer
        self.n_features = self._get_conv_output(self.input_shape)
        # Linear classifier, nn.Linear(in_features, out_features)
        self.classifier = nn.Linear(self.n_features, self.num_classes)

        if self.transfer:
            #Set into evaluation mode instead of training mode
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                # Turn off weight updating
                param.requires_grad = False

        # Returns the number of output features from the conv. layer
    def _get_conv_output(self, shape):
            batch_size = 1
            tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
            output_feat = self.feature_extractor(tmp_input) 
            n_size = output_feat.data.view(batch_size, -1).size(1)
            return n_size

    # Required overwrite function, flattens our output features then feeds then to our linear classifier
    def forward(self, x):
        #Pass data through our model
        x = self.feature_extractor(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classify
        x = self.classifier(x)
       
        return x

#============================================
# Custom CNN
# --------
# Class for creating a CNN, based off the in-class examples
#============================================

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
        for i, data in enumerate(train_loader, 0):
            # Move the data and classifications to the device
            inputs, classifications = data[0].to(device), data[1].to(device)
            # Clear gradients accumulated from previous steps
            optimizer.zero_grad()
            # Compute predicitions
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, classifications)
            # Backwards feedback for model parameters wrt loss
            loss.backward()
            # Update model parameters using computer optimizer gradients
            optimizer.step()

            train_loss += loss.item()
        print(f'{epoch + 1},  Train Loss: {train_loss / i:.3f},', end = ' ')

        # Update optimizer learning rate after each epoch
        scheduler.step()

        # Val is similar to train, but we dont need to compute the gradient
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, classifications = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, classifications)
                val_loss += loss.item()

            print(f'Val Loss: {val_loss / i:.3f}', end = ' ')

            # Model checkpointing, saves best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(model.state_dict(), BEST_MODEL_PATH + model_save_name)
                best_loss = val_loss

            end_time = time.time()
            print(f'\nEpoch Calculation Time (s): {(end_time - start_time):.1f}s')

    print("Training Completed")

#============================================
# Main
#============================================

print("#========================================================")
print("Script Test Info")
print("#========================================================")

print("Train set:\n", len(train_loader)*BATCH_SIZE)
print("\nVal set:\n", len(val_loader)*BATCH_SIZE)
print("\nTest set:\n", len(test_loader)*BATCH_SIZE)
print("\nTrain Loader Size:\n", len(train_loader))
print("\nVal Loader Size:\n", len(val_loader))
print("\nTest Loader Size:\n", len(test_loader))

train_batch = next(iter(train_loader))
print("\nBatch Information [#_samples, #_channels, x, y]:\n",train_batch[0].size())

# Init a device class
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nConnected Device:\n ", device)

class_names = train_dataset.classes
image_number = np.random.randint(low = 0, high = BATCH_SIZE - 1)
image_show(train_batch, image_number, class_names)
print("#========================================================")

# Init a model
resnet_model = resnetCNN(4, (3,224,224), True)
# Send our model to the device (GPU or CPU)
resnet_model.to(device)
nnTrainer(resnet_model, "resnet.pth", 0.001, 0.9, NUM_EPOCHS)
print("\n#========================================================")
