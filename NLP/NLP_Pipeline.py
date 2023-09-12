--------------------------------------------------------------------------------
Classical Sentiment Analysis:
--------------------------------------------------------------------------------
(1) Clean data removing special characters: keep only what can be useful for 
    the context;
(2) Tokenization
(3) POS tagging
(4) Lemmatizing: this will allow us to reduce our vocabulary. Use stemming if 
    no need for precision and speed is preferred.
(5) Remove stopwords: they won’t help a lot here. Doing it after POS Tagging can 
help to filter unwanted POS’s (keep adjectives, adverbs, verbs and nouns)
(6) Use BoW or TF-IDF (TF-IDF can supress the need for stopword removal, but large 
    vocabulary will be maintained).
(7) Apply your traditional Machine Learning Normalization Techniques.
The rest is default Machine Learning.

--------------------------------------------------------------------------------
Rule Based Information Extraction from Text:
--------------------------------------------------------------------------------
This is an activity that I find very frequently in StackOverflow nlp questions. 
It is related to be able to extract structured information from unstructured 
text input. The preprocessing pipeline is very simple, because we want to enjoy 
the most of the morphological features:

(1) Tokenization
(2) POS tagging
(3) Parsing: thats a step that we did not talk about — because I don’t consider 
it “preprocessing”, but rather core activity.
(4) NER: another step I don’t consider “preprocessing”.
→ Apply syntax/semantic rule matchers.

------------------------------------------
Smart Text Autocomplete/NLG:
------------------------------------------
(1) Tokenization
(2) Padding/Truncating
(3) Embedding
→ Train on RNN or Transformer model.

P.S.: In transformer models there’s almost no need for preprocessing, since 
the use of large quantity of documents surpass any input irregularity. 
Basically, only good embeddings are needed.

------------------------------------------------------------------
Classical Question Answering:
------------------------------------------------------------------
(1) Clean data removing special characters
(2) Tokenization
(3) POS tagging
(4) Lemmatizing
(5) BoW
Do the same preprocessing to both questions and answers. After that, 
compare the question vector to all answer vectors

