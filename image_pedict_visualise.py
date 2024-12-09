import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Reusing ForgeryDetectionCNN class from above
class ForgeryDetectionCNN(nn.Module):
    def __init__(self):
        super(ForgeryDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        feature_map = F.max_pool2d(x, 2)  # Extract feature map after the last pooling layer
        x = feature_map.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, feature_map

# Transformation for the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
def load_model(model_path):
    model = ForgeryDetectionCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Generate mask from feature maps
def generate_forgery_mask(feature_map, image_size):
    """
    Generate a mask from the feature map by summing up activations across channels
    and resizing it to the original image size.
    """
    # Aggregate feature map across the channel dimension
    mask = torch.sum(feature_map[0], dim=0).cpu().detach().numpy()
    
    # Normalize the mask to the range [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    # Resize the mask to the original image size
    mask = np.uint8(mask * 255)
    mask = Image.fromarray(mask).resize(image_size, resample=Image.BILINEAR)
    mask = np.array(mask) > 128  # Thresholding to create a binary mask
    return mask

# Display the image with forgery mask
def display_image_with_mask(image_path, mask):
    original_image = Image.open(image_path).convert('RGB')
    image_np = np.array(original_image)

    # Create a red overlay for the mask
    mask_overlay = np.zeros_like(image_np)
    mask_overlay[:, :, 0] = mask * 255  # Red channel for mask

    # Blend the original image and the mask
    blended_image = np.where(mask[:, :, None], mask_overlay, image_np)

    # Plot the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(blended_image)
    plt.axis('off')
    plt.title("Forgery Masked Image")

    plt.show()

# Predict and visualize forgery
def predict_and_visualize(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict using the model
    with torch.no_grad():
        outputs, feature_map = model(transformed_image)
        _, predicted = torch.max(outputs, 1)

    # Print prediction
    if predicted.item() == 0:
        print(f"The image {image_path} is classified as 'No Forgery'.")
    else:
        print(f"The image {image_path} is classified as 'Forgery'.")
        # Generate and display forgery mask
        mask = generate_forgery_mask(feature_map, image.size)
        display_image_with_mask(image_path, mask)

# Load the trained model
model_path = 'forgery_detection_model1.pth'
model = load_model(model_path)

# Provide the image path
image_path = 'dataset1/test/forgery/example_forgery_image.jpg'  # Replace with actual image path

# Predict and visualize
predict_and_visualize(model, image_path)
