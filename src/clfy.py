import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from  sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = "component_classification"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO gabe: function will return str of all the components
def get_dataset_component_path(root_dir):
    # example of how to get
    path = os.path.join(root_dir, "s*", "DSLR", "components", "**", "*.*")

    # array of all paths
    paths = glob.glob(path, recursive=True)

    return paths

class ImageLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths= get_dataset_component_path(root_dir)
        self.transform = transform
        self.labels = list(set([os.path.basename(p).split('_')[0] for p in self.image_paths]))
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = os.path.basename(img_path).split('_')[0]
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.label_to_idx[label]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageLabelDataset(root_dir='./dataset', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        ############################################
        self.bn1 = nn.BatchNorm2d(16)
        ############################################
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        ############################################
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        ############################################
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        ########################################################
        # Added Dropout Layer
        self.dropout = nn.Dropout(0.2)
        ########################################################
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        #######################################################
        x = self.pool(nn.functional.relu(self.conv3(x)))
        #######################################################
        # x = x.view(-1, 32 * 16 * 16)
        #######################################################
        x = x.view(-1, 64 * 8 * 8)
        #######################################################
        x = nn.functional.relu(self.fc1(x))
        ########################################################
        # Added Dropout Layer
        x = self.dropout(x)
        ########################################################
        x = self.fc2(x)
        return x

num_classes = len(dataset.labels)
model = CNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
###########################################################
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# Added weight_decay parameter 
# Penalizes larger weights in the model during training
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
###########################################################

###########################################################
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
##########################################################
# Training loop
def train_model(num_epochs=10, loss_plot_path="loss_curve.png"):
    best_acc = 0.0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    #define for early stoppage 
    patience = 15
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        acc = 100 * correct / total
        
        f1_prediction = predicted.numpy()
        f1_truth = labels.numpy()
        f1 = f1_score(f1_truth,f1_prediction,average='weighted')
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%, F1 Score: {f1}')

        #################################################
        # LEARNING RATE SCHEDULER
        scheduler.step(avg_val_loss)
        #################################################

        if avg_val_loss < best_val_loss:
            # print(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_mapping': dataset.label_to_idx
            }, f'{MODEL_NAME}.pth')
            # Reset early stopping counter
            counter = 0
        else:
            counter += 1
        
        # Also save if accuracy improves
        if acc > best_acc:
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_mapping': dataset.label_to_idx
            }, f'{MODEL_NAME}_best_acc.pth')
            best_acc = acc

        ################################################
        # Early stopping based on validation loss
        if counter >= patience:
            print("Early stopping triggered!")
            break
        ################################################

    # Plotting Losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(loss_plot_path)
    plt.show()



def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    '''transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # Basic augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        # Additional augmentations
        transforms.RandomVerticalFlip(),  # NEW: adds vertical flips
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # NEW: adds affine transformations
        # You can comment out individual augmentations to test their impact
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])'''


    checkpoint = torch.load(f'{MODEL_NAME}.pth', map_location=DEVICE)
    
    num_classes = len(checkpoint['label_mapping'])
    model = CNN(num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    idx_to_label = {v: k for k, v in checkpoint['label_mapping'].items()}
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities.data, 1)
        
        ret_arr = []

        for idx, t in enumerate(probabilities.flatten()):
            if t >= 0.7:
                #print(f'{idx_to_label[idx]}: {t}\n')
                ret_arr.append({idx_to_label[idx], t.item()})

        if len(ret_arr) == 0:
            ret_arr.append({idx_to_label[predicted.item()], confidence.item()})

    # return idx_to_label[predicted.item()], confidence.item()
    return ret_arr
