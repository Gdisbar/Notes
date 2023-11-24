import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define a simple denoising autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 1, 28, 28)  # Reshape to image size

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.2 * torch.randn_like(x))  # Add noise to the images
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and loss function
model = DenoisingAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training the denoising autoencoder
num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        images, _ = data
        noisy_images = images + 0.2 * torch.randn_like(images)  # Add noise to the images

        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the denoising autoencoder
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Plot original, noisy, and denoised images
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, _ = data
        noisy_images = images + 0.2 * torch.randn_like(images)
        denoised_images = model(noisy_images)

        plt.figure(figsize=(10, 3))

        for i in range(10):
            # Original image
            plt.subplot(3, 10, i + 1)
            plt.imshow(np.squeeze(images[i]), cmap='gray')
            plt.axis('off')

            # Noisy image
            plt.subplot(3, 10, i + 11)
            plt.imshow(np.squeeze(noisy_images[i]), cmap='gray')
            plt.axis('off')

            # Denoised image
            plt.subplot(3, 10, i + 21)
            plt.imshow(np.squeeze(denoised_images[i]), cmap='gray')
            plt.axis('off')

        plt.show()
        break  # Display one batch for brevity
