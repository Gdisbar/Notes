import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Embedding, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

# Sample data (replace with a larger dataset)
corpus = ["To be or not to be", "The Old Man and the Sea", "Romeo, Romeo, wherefore art thou?"]

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(corpus)
max_sequence_length = max(len(seq) for seq in input_sequences)
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

# VAE model
latent_dim = 2

# Encoder
inputs = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(total_words, latent_dim)(inputs)
flatten_layer = Flatten()(embedding_layer)
mean_layer = Dense(latent_dim)(flatten_layer)
log_var_layer = Dense(latent_dim)(flatten_layer)

# Reparameterization trick
def sampling(args):
    mean, log_var = args
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mean + K.exp(0.5 * log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([mean_layer, log_var_layer])

# Decoder
decoder_h = Dense(latent_dim, activation='relu')
decoder_mean = Dense(total_words, activation='softmax')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom loss function for VAE
xent_loss = binary_crossentropy(K.flatten(inputs), K.flatten(x_decoded_mean))
kl_loss = -0.5 * K.mean(1 + log_var_layer - K.square(mean_layer) - K.exp(log_var_layer), axis=-1)
vae_loss = xent_loss + kl_loss

# VAE Model
vae = Model(inputs, x_decoded_mean)
vae.add_loss(vae_loss)

# Compile the model
vae.compile(optimizer='rmsprop')
vae.summary()

# Train the model (replace with a more extensive dataset)
vae.fit(padded_sequences, epochs=100, batch_size=1)

# Generate text for a specific author
def generate_text_for_author(author_idx):
    # Sample from the learned distribution
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_sequence = decoder_mean.predict(random_latent_vector)[0]

    # Decode the generated sequence to text
    generated_text = tokenizer.sequences_to_texts([np.argmax(generated_sequence) + 1])[0]

    print(f"Generated text in the style of Author {author_idx + 1}: {generated_text}")

# Example: Generate text in the style of the first author
generate_text_for_author(0)


=================================
=================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Sample data (replace with a larger dataset)
corpus = ["To be or not to be", "The Old Man and the Sea", "Romeo, Romeo, wherefore art thou?"]

# Tokenize and pad sequences
word_to_index = {word: idx + 1 for idx, word in enumerate(set(" ".join(corpus).split()))}
index_to_word = {idx + 1: word for idx, word in enumerate(set(" ".join(corpus).split()))}

# Convert text to indices
indexed_sequences = [[word_to_index[word] for word in sentence.split()] for sentence in corpus]

# VAE model
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()

        self.embedding = nn.Embedding(input_size, latent_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)
        self.fc3 = nn.Linear(latent_size, input_size)

    def encode(self, x):
        x = self.flatten(self.embedding(x))
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return F.softmax(self.fc3(z), dim=-1)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Hyperparameters
input_size = len(word_to_index) + 1
latent_size = 2
learning_rate = 0.001
epochs = 100

# Model, optimizer, and loss
vae = VAE(input_size, latent_size)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    for sequence in indexed_sequences:
        sequence = torch.LongTensor(sequence).unsqueeze(0)  # Add batch dimension
        recon_batch, mu, log_var = vae(sequence)
        loss = torch.mean((F.cross_entropy(recon_batch, sequence.squeeze()) +
                           -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Generate text for a specific author
def generate_text_for_author(author_idx):
    random_latent_vector = torch.randn(1, latent_size)
    generated_sequence = vae.decode(random_latent_vector).detach().numpy()

    # Decode the generated sequence to text
    generated_text = " ".join([index_to_word[idx] for idx in generated_sequence.argmax(axis=1)])

    print(f"Generated text in the style of Author {author_idx + 1}: {generated_text}")

# Example: Generate text in the style of the first author
generate_text_for_author(0)
