import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from cnn import *

# Load the trained model
conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer
first_layer_weights = conv_net.conv1.weight.data.cpu()  # Shape: [out_channels, in_channels, kernel_h, kernel_w]

# Create a grid for kernel visualization
num_kernels = first_layer_weights.shape[0]
num_rows = int(np.ceil(np.sqrt(num_kernels)))
num_cols = int(np.ceil(num_kernels / num_rows))

# Create the kernel grid plot
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
for i in range(num_kernels):
    row = i // num_cols
    col = i % num_cols
    kernel = first_layer_weights[i, 0].numpy()  # Get the kernel for input channel 0
    
    # Normalize kernel values to [0, 1]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    
    if num_rows > 1:
        axes[row, col].imshow(kernel, cmap='gray')
        axes[row, col].axis('off')
    else:
        axes[col].imshow(kernel, cmap='gray')
        axes[col].axis('off')

# Remove empty subplots
for i in range(num_kernels, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    if num_rows > 1:
        axes[row, col].remove()

plt.tight_layout()
plt.savefig('kernel_grid.png', bbox_inches='tight', dpi=300)
plt.close()

# Load and process the sample image
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0                    # Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# Apply the convolution layer to get feature maps
with torch.no_grad():
    output = conv_net.conv1(img)

# Convert output to proper shape for plotting
output = output.squeeze(0)  # Remove batch dimension
output = output.cpu().numpy()  # Convert to numpy array

# Create grid for feature map visualization
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
for i in range(num_kernels):
    row = i // num_cols
    col = i % num_cols
    feature_map = output[i]
    
    # Normalize feature map values to [0, 1]
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    
    if num_rows > 1:
        axes[row, col].imshow(feature_map, cmap='gray')
        axes[row, col].axis('off')
    else:
        axes[col].imshow(feature_map, cmap='gray')
        axes[col].axis('off')

# Remove empty subplots
for i in range(num_kernels, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    if num_rows > 1:
        axes[row, col].remove()

plt.tight_layout()
plt.savefig('image_transform_grid.png', bbox_inches='tight', dpi=300)
plt.close()