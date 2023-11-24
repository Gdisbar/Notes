import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define a simple colorization autoencoder model
class ColorizationAutoencoder(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and loss function
model = ColorizationAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training the colorization autoencoder
num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        images, _ = data
        grayscale_images = images[:, 0:1, :, :]  # Take only one channel (grayscale)

        optimizer.zero_grad()
        outputs = model(grayscale_images)
        loss = criterion(outputs, grayscale_images)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the colorization autoencoder
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Plot original, grayscale, and colorized images
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, _ = data
        grayscale_images = images[:, 0:1, :, :]  # Take only one channel (grayscale)
        colorized_images = model(grayscale_images)

        plt.figure(figsize=(15, 5))

        for i in range(5):
            # Original image
            plt.subplot(3, 5, i + 1)
            plt.imshow(np.transpose(images[i], (1, 2, 0)))
            plt.axis('off')

            # Grayscale image
            plt.subplot(3, 5, i + 6)
            plt.imshow(np.squeeze(grayscale_images[i]), cmap='gray')
            plt.axis('off')

            # Colorized image
            plt.subplot(3, 5, i + 11)
            plt.imshow(np.transpose(colorized_images[i], (1, 2, 0)))
            plt.axis('off')

        plt.show()
        break  # Display one batch for brevity
