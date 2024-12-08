import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from predict_image import ForgeryDetectionCNN
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# Function to load the trained model
def load_model(model_path):
    model = ForgeryDetectionCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


# Grad-CAM: Generate heatmap
def generate_gradcam_heatmap(model, image_tensor, target_layer):
    gradients = []
    activations = []

    def save_activation_hook(module, input, output):
        activations.append(output)

    def save_gradient_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    target_layer.register_forward_hook(save_activation_hook)
    target_layer.register_full_backward_hook(save_gradient_hook)

    # Forward pass to get activations
    output = model(image_tensor)
    class_idx = torch.argmax(output).item()
    model.zero_grad()

    # Backward pass for the target class
    loss = output[0, class_idx]
    loss.backward()

    grad = gradients[0]
    activation = activations[0]

    # Process the gradients to obtain weights for each channel
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    weighted_activation = weights * activation

    # Sum the weighted activations across channels
    heatmap = torch.sum(weighted_activation, dim=1).squeeze().cpu().detach().numpy()

    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap

    return heatmap


# Function to overlay the heatmap on the original image
def overlay_heatmap(image, heatmap, alpha=0.6):
    img_array = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

    # Normalize the heatmap to fit the range [0, 255]
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Apply green color to the forged regions
    green_overlay = np.zeros_like(img_array)
    green_overlay[:, :, 1] = heatmap_resized  # Green channel

    # Create a mask for forged regions (regions with significant heatmap values)
    mask = heatmap_resized > 50  # Adjust threshold if needed

    # Only blur the non-forged parts
    blurred_image = cv2.GaussianBlur(img_array, (21, 21), 0)

    # Start with the original image and replace non-forged parts with the blurred image
    final_image = img_array.copy()
    final_image[~mask] = blurred_image[~mask]  # Only replace non-forged regions with blurred ones

    # Add the green overlay on the forged parts
    final_image[mask] = cv2.addWeighted(final_image, 1 - alpha, green_overlay, alpha, 0)[mask]

    return Image.fromarray(final_image)


# Function to predict if the image is a forgery or not
def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), image


# Function to highlight forged parts using Grad-CAM
def highlight_forgery(image_path, model, target_layer, transform):
    predicted_class, image = predict_image(model, image_path, transform)
    print(f"Predicted class (0 for no forgery, 1 for forgery): {predicted_class}")

    image_tensor = transform(image).unsqueeze(0)
    heatmap = generate_gradcam_heatmap(model, image_tensor, target_layer)

    # Highlight forged parts on the original image
    result_image = overlay_heatmap(image, heatmap)

    # Display the result
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

    # Save the result
    result_image.save("highlighted_forgery.png")


# Example usage
if __name__ == "__main__":
    # Load the trained model
    model = load_model('forgery_detection_model1.pth')

    # Define the transform for input image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Input image path
    image_path = 'input_image\Sp_D_CNN_A_ani0053_ani0054_0267.jpg'

    # Choose the target layer for Grad-CAM
    target_layer = model.conv3

    # Highlight forged parts
    highlight_forgery(image_path, model, target_layer, transform)
