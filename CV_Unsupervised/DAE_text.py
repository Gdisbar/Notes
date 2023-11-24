import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Sample data (replace with your own dataset)
corpus = ["To be or not to be", "The Old Man and the Sea", "Romeo, Romeo, wherefore art thou?"]

# Tokenize and add noise to create a noisy version
word_to_index = {word: idx + 1 for idx, word in enumerate(set(" ".join(corpus).split()))}
index_to_word = {idx + 1: word for idx, word in enumerate(set(" ".join(corpus).split()))}

# Convert text to indices
indexed_sequences = [[word_to_index[word] for word in sentence.split()] for sentence in corpus]

# Add noise to the data
noise_factor = 0.2
noisy_sequences = [[word_idx if torch.rand(1).item() > noise_factor else 0 for word_idx in sentence]
                   for sentence in indexed_sequences]

# Denoising Autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Embedding(input_size, hidden_size),
            nn.Flatten(),
            nn.Linear(hidden_size * len(indexed_sequences[0]), hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * len(indexed_sequences[0])),
            nn.ReLU(),
            nn.Unflatten(1, (len(indexed_sequences[0]), hidden_size)),
            nn.Embedding(hidden_size, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Hyperparameters
input_size = len(word_to_index) + 1
hidden_size = 64
learning_rate = 0.001
epochs = 100

# Model, optimizer, and loss
dae = DenoisingAutoencoder(input_size, hidden_size)
optimizer = optim.Adam(dae.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(epochs):
    for input_sequence, noisy_sequence in zip(indexed_sequences, noisy_sequences):
        input_tensor = torch.LongTensor(input_sequence).unsqueeze(0)
        noisy_tensor = torch.LongTensor(noisy_sequence).unsqueeze(0)

        optimizer.zero_grad()
        outputs = dae(noisy_tensor)
        loss = criterion(outputs.view(-1, input_size), input_tensor.view(-1))
        loss.backward()
        optimizer.step()

# Testing with a noisy sequence
test_noisy_sequence = noisy_sequences[0]
test_noisy_tensor = torch.LongTensor(test_noisy_sequence).unsqueeze(0)
reconstructed_tensor = dae(test_noisy_tensor)

# Decode the clean and noisy sequences
clean_sequence = [index_to_word[idx] for idx in indexed_sequences[0]]
reconstructed_sequence = [index_to_word[idx] for idx in reconstructed_tensor.squeeze().argmax(dim=1).tolist()]
noisy_sequence = [index_to_word[idx] for idx in test_noisy_sequence]

print("Clean Sequence:", clean_sequence)
print("Noisy Sequence:", noisy_sequence)
print("Reconstructed Sequence:", reconstructed_sequence)
