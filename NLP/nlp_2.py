# Let's covvert words to numbers using TF-IDF 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 10000)  # it contains only 10k features (fixed!)
xtrain_tfidf = vectorizer.fit_transform(xtrain).toarray()  # converting words to nos for train data 
xtest_tfidf = vectorizer.transform(xtest).toarray()        # converting words to numbers for test data 

-------------------------------------------------------------------------------------------------
def slicling_data(data):
    x = json.loads(data)
    try:
        len_title = len(x['title'])
    except:
        len_title = 0
    try:
    # selecting title and last 400-500 words from boilerplate
        split_text = x['body'].split(' ')[-500+len_title:]
    except:
        split_text=""

    if len_title:
        temp_text = ' '.join(split_text)+x['title']
    else:
        temp_text = ' '.join(split_text)

    return temp_text

# applies above func and stores result in a new column
df['sliced_data'] = df['boilerplate'].map(slicling_data)

temp = df['sliced_data'].map(lambda x: len(x))
temp[temp == 0]

import re
import string
import spacy
sp = spacy.load('en_core_web_sm')

def cleaning_text(text,emojis=True,html_tag=True,http=True,lemmitize=True,punctuation=True):
    
    #remove emojis
    if emojis is True:
        regrex_pattern = re.compile(pattern = "["                                                   
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                                       "]+", flags = re.UNICODE)

        text=regrex_pattern.sub(r'',text)
    #lower 
    text=text.lower()
    #remove html tag
    if html_tag is True:
        text=re.sub('<.*?>',"",text)
    #remove http link
    if http is True:
        text = re.sub("https?:\/\/t.co\/[A-Za-z0-9]*", '', text)
    #lemmitizing
    if lemmitize is True:
        lemmatized = [word.lemma_ for word in sp(text)]
        text = ' '.join(lemmatized)
    #remove punctuation
    if punctuation is True:
        text = text.translate(str.maketrans('', '', string.punctuation))
    # removing extra space
    text = re.sub("\s+", ' ', text)
    return text


df['sliced_data_cleaned']=df['sliced_data'].apply(lambda x: cleaning_text(x,lemmitize=False,
                                                                         http=False))


#================================================================================================
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics

def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# Kappa Scorer 
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

#==================================================================================================
# Avito Duplicate Ads Detection

pairs = pd.read_csv("../input/ItemPairs_train.csv", dtype=types1)
# Add 'id' column for easy merge
print("Load ItemInfo_train.csv")
items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)
items.fillna(-1, inplace=True)
location = pd.read_csv("../input/Location.csv")
category = pd.read_csv("../input/Category.csv") 

# Add text features --> length of title,description,AttrJSON
# merge dataframes for items1 & items2 --> add boolean columns,if they're equal or not

#===================================================================================================
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

most_common_tri = get_top_text_ngrams(df.text,10,3)
most_common_tri = dict(most_common_tri)
sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))
#===================================================================================================
stop_words = set(stopwords.words("english"))        
punctutions = set(string.punctuation)                
stop_words.update(puctuations)
news["text"] = news["text"].apply(lambda x : " ".join(x.lower() for x in x.split() if x not in stopwords))

news["text"] = news["text"].str.replace("[^\w\s]","")
news["text"] = news["text"].apply(lambda x : " ".join(re.sub(r"http\S+","",x) for x in x.split()))
news["text"] = news["text"].apply(lambda x : " ".join([Word(word).lemmatize() for word in x.split()]))

text_words = news[news["labels"]==1]["text"].str.split().apply(lambda x : [len(xx) for xx in x])
text_words.map(lambda x : np.mean(x))

#===================================================================================================
tokenizer = Tokenizer(num_words= 1000)
tokenizer.fit_on(X_train)
x_train_tokenized = tokenizer.text_to_sequences(X_train)
x_text_tokenized = tokenizer.text_to_sequences(X_test)
tokenizer.word_index.items()

def get_coef(word,*arr):
  return word,np.asarray(arr,dtype="float32")


embedding_index = dict(get_coef(*g.rstrip().rsplit()) for g in open("tweeter_embeddings"))
embedding = np.stack(embedding_index.values())
embedding_mean,embedding_std = np.mean(embedding),np.std(embedding)
embedding_size = embedding.shape[1]
word_index = tokenizer.word_index
nb_words = max(10000,len(word_index))

embedding_matrix = np.random.normal(embedding_mean,embedding_std,(nb_words,embedding_size))

for word, i in word_index.items():
    if i >= 10000:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model.add(Embedding(10000, output_dim=100, weights = [embedding_matrix], input_length=300, trainable = False ))
model.compile(),model.fit(),model.predict() # get np.argmax(axis=1)
