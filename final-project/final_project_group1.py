#============================================
# Group 1: Final Project
# ---------------------
# Tobin Eberle
# Jeff Wheeler
# Ryan Baker
# Tom Wilson
#============================================
# Stats for our datasets
# Kaggle/train_data: 79951 images
# susy/coco: 5001 images
# susy/midjourney: 3088
# susy/dalle3: 1657
# susy/realisticSDXL: 4202

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision
from torchvision import transforms

#============================================
# Datasets
#============================================
class KaggleDataset(Dataset):
    def __init__(self, transform, dir):
        self.transform = transform
        self.dir = dir
        self.images = np.empty(0)
        self.labels = np.empty(0)
    
        #Load labels
        label_df = pd.read_csv(os.path.join(self.dir + "/train.csv"))
        def label_from_df(image_name):
            return label_df.loc[label_df['file_name'] == image_name, 'label'].item()

        #Load images
        def load_images():
            class_path = os.path.join(self.dir, "train_data")
            images = os.listdir(class_path)
            for image in images:
                img_path = os.path.join(class_path, image)
                if os.path.isfile(img_path):
                    self.images = np.append(self.images, img_path)
                    self.labels = np.append(self.labels, label_from_df("train_data/" + image))
        load_images()
                        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Need to return the image as tensor
        image = torchvision.io.decode_image(image, mode= "RGB")
        image = transforms.functional.to_pil_image(image)
        image = self.transform(image)

        # Note: 0 is real, 1 is AI
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

class SUSYDataset(Dataset):
    def __init__(self, transform, dir):
        self.transform = transform
        self.dir = dir
        self.images = np.empty(0)
        self.labels = np.empty(0)
        self.names = np.empty(0)

        #Load images
        def load_images():
            classes = sorted([label for label in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, label))])
            label_map = {class_name: idx for idx, class_name in enumerate(classes)}
            
            for class_name in classes:
                class_path = os.path.join(self.dir, class_name, class_name + "_dataset")
                if os.path.isdir(class_path):
                    images = os.listdir(class_path)
                    for image in images:
                        img_path = os.path.join(class_path, image)
                        if os.path.isfile(img_path):
                            self.images = np.append(self.images, img_path)
                            self.labels = np.append(self.labels, label_map[class_name])
                            self.names = np.append(self.names, class_name)
                            
        load_images()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Need to return the image as tensor
        image = torchvision.io.decode_image(image, mode= "RGB")
        image = transforms.functional.to_pil_image(image)
        image = self.transform(image)

        return {
            # Verify labels using the name
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'name' : self.names[idx]
        }

if __name__ == "__main__":

    KAGGLE_PATH = "/work/TALC/enel645_2025w/group1-dataset/kaggle_real_ai_dataset" 
    SUSY_PATH = "/work/TALC/enel645_2025w/group1-dataset/susy"

    # Transforms
    data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    # Create datasets
    # Note, need to remove the enumerate() from the datasets in order to load the whole things
    kaggle_dataset = KaggleDataset(data_transforms, KAGGLE_PATH)
    # susy_dataset = SUSYDataset(data_transforms, SUSY_PATH)

    # View example images
    kaggle_dataloader = DataLoader(kaggle_dataset, batch_size = 3, shuffle = True)
    # susy_dataloader = DataLoader(susy_dataset, batch_size = 3, shuffle = True)
   
   # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    k_batch = next(iter(kaggle_dataloader))
    s_batch = next(iter(susy_dataloader))
    for i in range(3):
         img = k_batch['image'].numpy()[i].transpose(1,2,0)
         label = k_batch['label'].numpy()[i]
         axes[0,i].imshow(img)
         axes[0,i].set_title(f'Label: {label}')
         axes[0,i].axis('off')
    # for i in range(3):
    #      img = s_batch['image'].numpy()[i].transpose(1,2,0)
    #      label = s_batch['label'].numpy()[i]
    #      name = s_batch['name'][i]
    #      axes[1,i].imshow(img)
    #      axes[1,i].set_title(f'Label: {name}')
    #      axes[1,i].axis('off')

    fig.text(0.04, 0.75, 'Kaggle Images', va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.text(0.04, 0.25, 'Susy Images', va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')
    plt.savefig("test-images.png")
    plt.show()

    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = torch.utils.random_split(kaggle_dataset, [0.7, 0.2, 0.1])
    torch.save(train_dataset, "./train.pt")
    torch.save(val_dataset, "./val.pt")
    torch.save(test_dataset, "./test.pt")

    # train_loader = DataLoader(train_dataset, batch_size = 3, shuffle = True)
    # val_loader = DataLoader(val_dataset, batch_size = 3, shuffle = True)
    # test_loader = DataLoader(test_dataset, batch_size = 3, shuffle = False)




