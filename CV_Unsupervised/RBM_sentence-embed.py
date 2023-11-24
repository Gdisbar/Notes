import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# Sample text data
sentences = [
    "This is an example sentence.",
    "Sentence embeddings are useful.",
    "RBM can be used for feature learning."
]

# Convert text data to binary feature vectors using CountVectorizer
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(sentences).toarray()

# Build an RBM for feature learning
rbm = BernoulliRBM(n_components=50, learning_rate=0.01, n_iter=20)
X_features = rbm.fit_transform(X)

# Normalize the RBM features to get sentence embeddings
sentence_embeddings = normalize(X_features, norm='l2')

# Display the sentence embeddings
for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Sentence Embedding: {sentence_embeddings[i]}")
    print()


=================================
=================================

import numpy as np
from gensim.models import Word2Vec
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize

# Sample text data
sentences = [
    "This is an example sentence.",
    "Sentence embeddings are useful.",
    "RBM can be used for feature learning."
]

# Load pre-trained Word2Vec embeddings (you can replace this with GloVe if needed)
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)

# Function to get sentence embeddings using Word2Vec
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Get Word2Vec embeddings for each sentence
sentence_embeddings = [get_sentence_embedding(sentence, word2vec_model) for sentence in sentences]

"""
# Load pre-trained GloVe embeddings using spaCy
nlp = spacy.load("en_core_web_md")

# Function to get sentence embeddings using GloVe
def get_sentence_embedding(sentence, nlp):
    doc = nlp(sentence)
    vectors = [token.vector for token in doc]
    if not vectors:
        return np.zeros(nlp.vocab.vectors.shape[1])
    return np.mean(vectors, axis=0)


"""
# Normalize the embeddings
sentence_embeddings = normalize(sentence_embeddings, norm='l2')

# Build an RBM for further feature learning (similar to the previous example)
rbm = BernoulliRBM(n_components=50, learning_rate=0.01, n_iter=20)
X_features = rbm.fit_transform(sentence_embeddings)

# Display the sentence embeddings
for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Sentence Embedding: {X_features[i]}")
    print()
    
=================================
=================================



=================================
=================================
