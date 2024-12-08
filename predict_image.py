import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn  # Importing nn for layers like Conv2d, Linear, etc.
import torch.nn.functional as F  # Importing functional for activation functions like ReLU
import os

# Define the transform for input image (same as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(30),  # Randomly rotate the image up to 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Randomly crop and resize the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model class again (same as in your previous code)
class ForgeryDetectionCNN(nn.Module):
    def __init__(self):
        super(ForgeryDetectionCNN, self).__init__()
        
       # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization after conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization after conv2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization after conv3
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes: forgery or no forgery)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2)
        #x = F.relu(self.conv3(x))
        #x = F.max_pool2d(x, 2)
        x = F.relu(self.bn1(self.conv1(x)))  # Apply batch norm before activation
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load the trained model
def load_model(model_path):
    model = ForgeryDetectionCNN()  # Create a new instance of the model
    model.load_state_dict(torch.load(model_path,weights_only=True))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict if the image is a forgery or not
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')  # Open and convert the image to RGB
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make the prediction
    with torch.no_grad():  # Disable gradient computation during inference
        outputs = model(image)  # Forward pass
        
        # Print the raw output (logits)
        print(f"Raw output (logits) for {image_path}: {outputs}")
        
        # Get the predicted class with the highest score
        _,predicted = torch.max(outputs, 1)
        # _,predicted = torch.max(outputs)

        print(f"Predicted class (0 for no forgery, 1 for forgery): {predicted.item()}")
        #image_path = 'input_image\80.tif'
        #predict_image(model, image_path)
    



# Example usage:
# model is your trained model
# image_path is the path to the image you want to classify
# transform is the preprocessing function applied to the image (e.g., resizing, normalization)


    # Interpret the prediction
    if predicted.item() == 0:
        print(f"The image {image_path} is classified as 'No Forgery'.")
    else:
        print(f"The image {image_path} is classified as 'Forgery'.")
   

# Load the trained model (you need to specify the correct path to your saved model)
model = load_model('forgery_detection_model1.pth')

# Provide the path to the image you want to classify
# For example, if your image is in the same directory as the script, use:
image_path = 'input_image\Sp_D_CND_A_pla0005_pla0023_0281.jpg'

# Make a prediction on the image
predict_image(model, image_path)
