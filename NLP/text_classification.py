nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
------------------------------------------------------------------------------------------------
vect = CountVectorizer(preprocessor=clean, max_features=5000) # limit noise
logreg = LogisticRegression(class_weight="balanced") # handle class imbalance
-------------------------------------------------------------------------------------------------
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

train_data, test_data, train_cats, test_cats = train_test_split(mydata,mycats,random_state=1234)

train_doc2vec = [TaggedDocument((d), tags=[str(i)]) for i, d in enumerate(train_data)]

model = Doc2Vec(vector_size=50, alpha=0.025, min_count=5, dm =1, epochs=100) # tweet representations
model.build_vocab(train_doc2vec)
model.train(train_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)
model.save("d2v.model")

model= Doc2Vec.load("d2v.model")  #Infer the feature representation
#infer in multiple steps to get a stable representation. 
train_vectors =  [model.infer_vector(list_of_tokens, steps=50) for list_of_tokens in train_data]
test_vectors = [model.infer_vector(list_of_tokens, steps=50) for list_of_tokens in test_data]
-------------------------------------------------------------------------------------------------------------
# Creating a feature vector by averaging all embeddings for all sentences
def embedding_feats(list_of_lists):
    DIMENSION = 300
    zero_vector = np.zeros(DIMENSION)
    feats = []
    for tokens in list_of_lists: # each sentence of text
        feat_for_this =  np.zeros(DIMENSION)
        count_for_this = 0 + 1e-5 # to avoid divide-by-zero 
        for token in tokens:  # each token of the sentence
            if token in w2v_model:
                feat_for_this += w2v_model[token]
                count_for_this +=1
        if(count_for_this!=0):
            feats.append(feat_for_this/count_for_this) 
        else:
            feats.append(zero_vector)
    return feats


train_vectors = embedding_feats(texts_processed) # 300 dim feature vector for each sentence

from gensim.models import Word2Vec, KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format(path_to_model, binary=True) #GoogleNews-vectors-negative300.bin
word2vec_vocab = w2v_model.vocab.keys()
word2vec_vocab_lower = [item.lower() for item in word2vec_vocab]


------------------------------------------------------------------------------------------------------------
text = text.normalize('NFKD').str.encode('ascii','ignore').str.decode('utf-8')

from fasttext import train_supervised 
model = train_supervised(input=train_file, label="__class__", lr=1.0, epoch=75, loss='ova', wordNgrams=2, dim=200, thread=2

-------------------------------------------------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS) 
tokenizer.fit_on_texts(train_texts) # no need to fit on test_tests
train_sequences = tokenizer.texts_to_sequences(train_texts) #Converting text to a vector of word indexes 
word_index = tokenizer.word_index 
trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH) # same for test_sequence
trainvalid_labels = to_categorical(np.asarray(train_labels))

# split the training data into a training set and a validation set
indices = np.arange(trainvalid_data.shape[0])
np.random.shuffle(indices)
trainvalid_data = trainvalid_data[indices]
trainvalid_labels = trainvalid_labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * trainvalid_data.shape[0])

# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# prepare embedding matrix - rows are the words from word_index, columns are the embeddings of that word from glove.
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load these pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,trainable=False)

-------------------------------------------------------------------------------------------------------------------------
def strip(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub('\[[^]]*\]', '', soup.get_text())
    pattern=r"[^a-zA-z0-9\s,']"
    text=re.sub(pattern,'',text)
    return text
sentence = ["[CLS] "+i+" [SEP]" for i in sentences]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:510] , sentence))

# Set the maximum sequence length. 
MAX_LEN = 128

# Pad our input tokens so that everything has a uniform length
input_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, tokenized_texts)),
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# same for validation
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)#binary classification
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
------------------------------------------------------------------------------------------------------------------------
from ktrain import text

c = make_pipeline(vect, classifier)
mystring = list(X_test)[221] #Take a string from test instance
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(mystring, c.predict_proba, num_features=6)
-------------------------------------------------------------------------------------------------------------------------
class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list 
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self
    
    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))
        
sequencer = TextsToSequences(num_words=vocab_size)




class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length. 
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros
    
    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during 
        transform it is transformed to a 0
    """
    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None
        
    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self
    
    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X

padder = Padder(maxlen)

# Use Keras Scikit-learn wrapper to instantiate a LSTM with all methods
# required by Scikit-learn for the last step of a Pipeline
sklearn_lstm = KerasClassifier(build_fn=create_model, epochs=2, batch_size=32, 
                               max_features=max_features, verbose=1)

# Build the Scikit-learn pipeline
pipeline = make_pipeline(sequencer, padder, sklearn_lstm)
# Build the Scikit-learn pipeline
pipeline = make_pipeline(sequencer, padder, sklearn_lstm)
explainer = LimeTextExplainer(class_names=class_names)
explanation = explainer.explain_instance(text_sample, pipeline.predict_proba, num_features=10)

weights = OrderedDict(explanation.as_list())
lime_weights = pd.DataFrame({'words': list(weights.keys()), 'weights': list(weights.values())})
-------------------------------------------------------------------------------------------------------------------
  # we use the first 100 training examples as our background dataset to integrate over
explainer = shap.DeepExplainer(rnnmodel, x_train[:20])

# explain the first 10 predictions
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(x_val[:5])

import numpy as np
words = imdb.get_word_index()
num2word = {}
for w in words.keys():
    num2word[words[w]] = w
x_val_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_val[i]))) for i in range(10)])
----------------------------------------------------------------------------------------------------------------
df_train, df_test = train_test_split(df,stratify = df['target'], test_size = 0.2, random_state = 2020)
# Language model data 
data_lm = TextLMDataBunch.from_df(train_df = df_train, valid_df = df_test, path = "")  
# Classifier model data 
data_class = TextClasDataBunch.from_df(path = "", train_df = df_train, valid_df = df_test, vocab=data_lm.train_ds.vocab, bs=32)
model = language_model_learner(data_lm,  arch = AWD_LSTM, pretrained = True, drop_mult=0.5)
model.fit_one_cycle(4, max_lr= 5e-02)#you can freeze and unfreeze different layers and by doing so we can have different lr for each layer
#for freezing and unfreezing code you can refer https://docs.fast.ai/text.html
for i in range(10):
    print(model.predict("The food is", n_words=15))
model.lr_find() # you can find more details about this at https://docs.fast.ai/basic_train.html
model.recorder.plot(suggestion=True)
