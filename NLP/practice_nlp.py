# remove punctuation from text --> return as a list of words
sents = " ".join([sent for sent in sentences if sent not in stopwords_])
text = " ".join([sent for sent in sents.lower() if sent not in punctuations])

X = scipy.sparse.hstack((q1,q2))
tf_vect = TfidfVectorizer(analyzer="word",token_pattern=r'w\{1,}',)

texts = ' '.join(map(lambda x : x.text,soup.findall('p')))
sentences = []
for text in texts:
	sentences.append(sent_tokenize(text))
sentences = [sent for sentence in sentences if sent in sentence]
# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]"," ")
clean_sentences = [remove_stopwords(s.split())for s in clean_sentences]
word_embeddings = {}
with open("glove_6B_100d.txt",encoding="utf-8") as f:
	for line in f:
		value = line.split()
		word = value[0]
		embed = np.asarray(value[1:],dtype="float32")
		word_embeddings[words] = embed

sentence_vectors= []
embed_dim = 100
for sentence in clean_sentences:
	if len(sentence)!=0:
		v = sum(word_embeddings.get(w,np.zeros(embed_dim,))
			for w in sentence.split())/(len(sentence.split())+0.001)
	else:
		v = np.zeros(embed_dim,)

	sentence_vectors.append(v)

# Create an empty similarity matrix
sim_mat = np.zeros([len(sentences),len(sentences)])
for i in range(len(sentences)):
	for j in range(len(sentences)):
		if i!=j:
			sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,embed_dim),
								sentence_vectors[j].reshape(1,embed_dim))[0,0]


(mu,sig) = norm.fit(np.log1p(df["price"]))
sns.distplot(np.log1p(df["price"]),fit=norm),stats.probplot(np.log1p(df["price"]),fit=plot)

preprocess = ColumnTransformer([
				("onehot_item_category",OneHotEncoder()["item_category"]),
				("countvect_name",CountVectorizer(min_df=MIN_DF),"name"),
				("tfidf_description",TfidfVectorizer(max_features=MAX_FEATURE,stopwords="english",
					ngram=(1,3)),"description")
	])

model = make_pipeline(preprocess,Ridge(solver="lsrq",fit_intercept=False))
model.fit(x_train,y_train)

## use different ngram to get single word / word phrases
def get_topn_words(corpus,n=1):
	vec = CountVectorizer(stopwords="english",ngram=(n,n)).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_of_words = bag_of_words.sum(axis=0)
	freq_words = [(word,sum_of_words[0,idx] for word,idx in vec.vocabulary_.item())]
	freq_words = sorted(freq_words,key=lambda x : x[1],resversed=True)
	return freq_words[:n]


word_freq = get_topn_words("This is a test corpus to check how it works",3)
for word,freq in word_freq:
	print(word,freq)

nlp = spacy.load("de",disable=["ner","parser"])
def lemmatizer(text):
	sent = []
	doc = nlp(text)
	for word in doc:
		sent.append(word.lemma_)
	return " ".join(sent)

df["comment_lemmatize"] = df.apply(lambda x : lemmatizer(x["comment"]),axis=1)
df["comment_lemmatize"] = df["comment_lemmatize"].str.replace("-PRON-","")
sentences = [row.split() for row in  df["comment_lemmatize"]]