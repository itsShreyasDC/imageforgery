import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# CNN Model definition
class ForgeryDetectionCNN(nn.Module):
    def __init__(self):
        super(ForgeryDetectionCNN, self).__init__()
        
        # Convolutional layers (kernel size 3x3, padding 1, stride 1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization after conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization after conv2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization after conv3
        
        # After 3 convolutional layers and 3 pooling layers, the image size is reduced to 16x16.
        # The output size before the fully connected layer is 64 * 16 * 16 = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer (Input: 64 * 16 * 16 = 16384, Output: 128)
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes: forgery or no forgery)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Apply batch norm before activation
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor to 1D for the fully connected layer
        x = F.relu(self.fc1(x))    # First fully connected layer + ReLU
        x = self.fc2(x)            # Second fully connected layer (output)
        return x

# Dataset class to load images and labels
class ForgeryDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Path to the image directory.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load the image paths and labels for 'forgery' and 'no_forgery' classes
        for label in ['forgery', 'no_forgery']:
            class_dir = os.path.join(image_dir, label)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):  # Only consider .tif images
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(0 if label == 'no_forgery' else 1)  # Label 0 for 'no_forgery', 1 for 'forgery'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformation (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(30),  # Randomly rotate the image up to 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Randomly crop and resize the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the datasets
train_dataset = ForgeryDataset(image_dir='dataset1/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ForgeryDataset(image_dir='dataset1/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = ForgeryDetectionCNN()
criterion = nn.CrossEntropyLoss()  # Loss function (cross entropy)
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Optimizer (Adam)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute the running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Evaluate the model
model.eval()
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

test_accuracy = correct_preds / total_preds * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'forgery_detection_model1.pth')

# Load the trained model
model.load_state_dict(torch.load('forgery_detection_model1.pth', weights_only=True))
