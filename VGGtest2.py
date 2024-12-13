import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def plot_pca(pca_image: np.ndarray, output_size: tuple, save_dir: str, save_prefix: str = ''):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Normalize the pca_image
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
    pca_image = (pca_image * 255).astype(np.uint8)

    # Apply color map
    heatmap_color = cv2.applyColorMap(pca_image, cv2.COLORMAP_JET)

    # Resize heatmap_color to match output_size
    heatmap_color = cv2.resize(heatmap_color, output_size[::-1])

    # Save the heatmap_color if necessary
    if save_prefix:
        cv2.imwrite(str(save_dir / f"{save_prefix}frame210.png"), heatmap_color)

    return heatmap_color


# Load VGG11 model
vgg11 = models.vgg11(pretrained=True)
vgg11.eval()

# Load and preprocess the image
image_path = "data/demos/frame210.jpg"
image = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Get the features192.168.1.105
with torch.no_grad():
    features = vgg11.features(input_batch)

# Compute the average of each feature map
avg_pooled_features = torch.mean(features, dim=1, keepdim=True)

# Convert features to numpy array
features_np = avg_pooled_features.squeeze(0).numpy()

# Define the output size
output_size = (448, 448)

# Plot the heat map using plot_pca function with specified output size
heatmap_color = plot_pca(features_np[0], output_size, "/home/wubin/code/SamPose", "feature_map")
