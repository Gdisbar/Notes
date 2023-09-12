SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def clean(text, stem_words=True):
    import re
    from string import punctuation
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

#    stops = set(stopwords.words("english"))
    # Clean the text, with the option to stem words.
    
    # Empty question
    
    if type(text) != str or text=='':
        return ''

    # Clean the text
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
#     # all numbers should separate from words, this is too aggressive
    
#     def pad_number(pattern):
#         matched_string = pattern.group(0)
#         return pad_str(matched_string)
#     text = re.sub('[0-9]+', pad_number, text)
    
    # add padding to punctuations and special chars, we still need them later
    
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
#    def pad_pattern(pattern):
#        matched_string = pattern.group(0)
#       return pad_str(matched_string)
#    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text) 
        
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # replace non-ascii word with special word
    
    # indian dollar
    
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text) 
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"
    
    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
  
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
       # Return a list of words
    return text

X = scipy.sparse.hstack((trainq1_trans,trainq2_trans)) ## fitted using CountVectorizer()
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

-----------------------------------------------------------------------------------------------------
=====================================================================================================
                            ### Summarization of speech - scrapped from web
=====================================================================================================
-----------------------------------------------------------------------------------------------------



sentences = []
for s in text:    # text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]

# ['An amazing quality of life for all of our citizens is within reach.',
#  'We can make our communities safer, our families stronger, our culture richer, our faith deeper, and our middle class bigger and more prosperous than ever before.',
#  '(Applause.)',
#  'But we must reject the politics of revenge, resistance, and retribution, and embrace the boundless potential of cooperation, compromise, and the common good.',
#  '(Applause.)',
#  'Together, we can break decades of political stalemate.',
#  'We can bridge old divisions, heal old wounds, build new coalitions, forge new solutions, and unlock the extraordinary promise of America’s future.',
#  'The decision is ours to make.',
#  'We must choose between greatness or gridlock, results or resistance, vision or vengeance, incredible progress or pointless destruction.',
#  'Tonight, I ask you to choose greatness.']

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)

# Create an empty similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 15 sentences as the summary representation
for i in range(10):
    print(ranked_sentences[i][1])

# After 24 months of rapid progress, our economy is the envy of the world, our military is the most powerful on Earth, by far, and America — (applause) — America is again winning each and every day.
# Millions of our fellow citizens are watching us now, gathered in this great chamber, hoping that we will govern not as two parties but as one nation.
# I hope you can pass the USMCA into law so that we can bring back our manufacturing jobs in even greater numbers, expand American agriculture, protect intellectual property, and ensure that more cars are proudly stamped with our four beautiful words: “Made in the USA.”  (Applause.)
# When Grace completed treatment last fall, her doctors and nurses cheered — they loved her; they still love her — with tears in their eyes as she hung up a poster that read: “Last day of chemo.”  (Applause.)
# We do not know whether we will achieve an agreement, but we do know that, after two decades of war, the hour has come to at least try for peace.
# Perhaps — (applause) — we really have no choice.
# And we must always keep faith in America’s destiny that one nation, under God, must be the hope and the promise, and the light and the glory, among all the nations of the world.
# Tonight, I am also asking you to join me in another fight that all Americans can get behind: the fight against childhood cancer.
# Lawmakers in New York cheered with delight upon the passage of legislation that would allow a baby to be ripped from the mother’s womb moments from birth.
# Judah says he can still remember the exact moment, nearly 75 years ago, after 10 months in a concentration camp, when he and his family were put on a train and told they were going to another camp.

=====================================================================================================
##### from gensim.summarization import summarize
##### from gensim.summarization import keywords

-----------------------------------------------------------------------------------------------------
=====================================================================================================
                        #####    
=====================================================================================================
-----------------------------------------------------------------------------------------------------
sns.distplot(np.log1p(df['price']), fit = norm);
(mu, sigma) = norm.fit(np.log1p(df['price']))
res = stats.probplot(np.log1p(df['price']), plot=plt)

pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

preprocess = ColumnTransformer(
    [('item_condition_category', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['item_condition_id']),
     ('brand_name_category', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['brand_name']),
     ('category_name_countvec', CountVectorizer(), 'category_name'),
     ('name_countvec', CountVectorizer(min_df=NAME_MIN_DF), 'name'),
     ('description_tfidf', TfidfVectorizer(max_features = MAX_FEAT_DESCP, stop_words = 'english', ngram_range=(1,3)), 'item_description')],
    remainder='passthrough')

model = make_pipeline(preprocess,Ridge(solver = "lsqr", fit_intercept=False))
model.fit(X_train, y_train)

# Pipeline(memory=None,
#      steps=[('columntransformer', ColumnTransformer(n_jobs=None, remainder='passthrough', sparse_threshold=0.3,
#          transformer_weights=None,
#          transformers=[('item_condition_category', OneHotEncoder(categorical_features=None, categories=None, dtype='int',
#        handle_unknown='ignore', n_va...t_intercept=False, max_iter=None,
#    normalize=False, random_state=None, solver='lsqr', tol=0.001))])

=====================================================================================================
df['state_abbrev'] = df['state'].map(us_state_abbrev).fillna(df['state'])

# use different ngram_range to get single word(default) & sub-words(phrases)
def get_top_n_words(corpus, n=None): 
    vec = CountVectorizer(ngram_range=(2, 2),stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df['comment'], 20)
for word, freq in common_words:
    print(word, freq)

df['comment_polarity'] = df['comment'].map(lambda text: TextBlob(text).sentiment.polarity)
nums_polarity = df.query('comment_polarity != 1000')['comment_polarity']
fig = ff.create_distplot(hist_data = [nums_polarity], group_labels = ['Comment polarity'])
fig.update_layout(title_text='Distribution of sentiment polarity in comment', template="plotly_white")
fig.show()

nlp = spacy.load('en', disable=['ner', 'parser']) 

def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

df_clean["comment_lemmatize"] =  df_clean.apply(lambda x: lemmatizer(x['comment']), axis=1)
df_clean['comment_lemmatize'] = df_clean['comment_lemmatize'].str.replace('-PRON-', '')

sentences = [row.split() for row in df_clean['comment_lemmatize']]
keywords = ["economy", "health", "people", "virus","need", "work", "testing", "friedman",
            "risk", "medical", "care", 'infect', 'president', 'approach', 'week', 'know']
words = [word for word in keywords if word in list(w2v_model.wv.vocab)]
X = model_wv_df.T[words].T
pca = PCA(n_components=2)
result = pca.fit_transform(X)
df = pd.DataFrame(result, columns=["Component 1", "Component 2"])
df["Word"] = keywords
df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2)

-----------------------------------------------------------------------------------------------------
=====================================================================================================
                        #####  Dedupe
=====================================================================================================
-----------------------------------------------------------------------------------------------------

df.loc[df['name'] == 'Roy Street Commons']

#                 name  |  address
# ----------------------|---------------------------------------------
# 82  Roy Street Commons|  621 12th Ave E, Seattle, WA 98102
# 90  Roy Street Commons|  621 12th Avenue East, Seattle, Washington 98102

import sparse_dot_topn.sparse_dot_topn as ct
from scipy.sparse import csr_matrix

def awesome_cossim_top(A, B, ntop, lower_bound=0):
  
    A = A.tocsr() # compressed sparse row-format
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 5)

# <168x168 sparse matrix of type '<class 'numpy.float64'>'
#     with 840 stored elements in Compressed Sparse Row format>

def get_matches_df(sparse_matrix, name_vector, top=840):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similarity': similairity})

matches_df = get_matches_df(matches, name_address)

matches_df[matches_df['similarity'] < 0.99999].sort_values(by=['similarity'], ascending=False).head(30)

#                                             left_sid   |                                     right_side    |   similarity
# -------------------------------------------------------|---------------------------------------------------|---------------
# 826 Pike''s Place Lux Suites by Barsala 2nd Ave and... |  Pike's Place Lux Suites by Barsala 2rd Ave and...|   0.715406
# 831 Pike''s Place Lux Suites by Barsala 2rd Ave and... |  Pike's Place Lux Suites by Barsala 2nd Ave and...|   0.715406

-----------------------------------------------------------------------------------------------------
=====================================================================================================
                        #####  Doc2Vec
=====================================================================================================
-----------------------------------------------------------------------------------------------------
#                                             narrative | Product
# ------------------------------------------------------|---------------------------------------
# 0   when my loan was switched over to navient i wa... |  Student loan
# 1   i tried to sign up for a spending monitoring p... |  Credit card or prepaid card