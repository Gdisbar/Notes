import torch
import torch.nn as nn
import torch.optim as optim
from Bio import Entrez, SeqIO

# Set your email address for Entrez
Entrez.email = "your_email@example.com"

# Function to fetch a DNA sequence from GenBank
def fetch_dna_sequence(accession):
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return str(record.seq)

# Fetch a DNA sequence from GenBank (replace with your accession number)
accession_number = "NM_001301717"  # Example accession number
dna_sequence = fetch_dna_sequence(accession_number)

# Convert DNA sequence to one-hot encoding
def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return [mapping[nucleotide] for nucleotide in sequence]

# Convert the fetched sequence to one-hot encoding
one_hot_sequence = [one_hot_encoding(dna_sequence)]

# Convert to PyTorch tensor
data_tensor = torch.FloatTensor(one_hot_sequence)

# Recurrent Autoencoder model (same as before)
class RecurrentAutoencoder(nn.Module):
    # ... (unchanged)

# Model, optimizer, and loss
rae = RecurrentAutoencoder(input_size=4, hidden_size=8)
optimizer = optim.Adam(rae.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training (unchanged)
for epoch in range(100):
    optimizer.zero_grad()
    reconstructed_data = rae(data_tensor)
    loss = criterion(reconstructed_data, data_tensor)
    loss.backward()
    optimizer.step()

# Reconstruct the original sequence
reconstructed_sequence = rae(data_tensor)

# Convert the reconstructed sequence back to nucleotides
reconstructed_sequence = [max('ACGT', key=lambda x: nucleotide) for nucleotide in reconstructed_sequence.squeeze().detach().numpy()]

# Print the original and reconstructed sequences
print("Original Sequence:")
print(dna_sequence)
print("\nReconstructed Sequence:")
print(''.join(reconstructed_sequence))
