

# #Get one hot representation for any string based on this vocabulary. 
# #If the word exists in the vocabulary, its representation is returned. 
# #If not, a list of zeroes is returned for that word. 
# def get_onehot_vector(somestring):
#     onehot_encoded = []
#     for word in somestring.split():
#         temp = [0]*len(vocab)
#         if word in vocab:
#             temp[vocab[word]-1] = 1 # -1 is to take care of the fact indexing in array starts from 0 and not 1
#         onehot_encoded.append(temp)
#     return onehot_encoded


#vocab = {'dog': 1, 'bites': 2, 'man': 4, 'eats': 2, 'meat': 5, 'food': 3}
#vocab = {'dog': 1, 'bites': 0, 'man': 4, 'eats': 2, 'meat': 5, 'food': 3}
# vocab = {'dog': 3, 'bites': 2, 'man': 5, 'and': 0, 'are': 1, 'friends': 4}
# one_hot_encoded = get_onehot_vector("dog bites man") 
# print(one_hot_encoded)
# # [[1, 0, 0, 0, 0, 0], 
# # [0, 0, 0, 0, 0, 1], 
# # [0, 0, 0, 1, 0, 0]]

# # [[1, 0, 0, 0, 0, 0], 
# # [0, 1, 0, 0, 0, 0], 
# # [0, 0, 0, 1, 0, 0]]


# [[0, 0, 1, 0, 0, 0], 
# [0, 1, 0, 0, 0, 0], 
# [0, 0, 0, 0, 1, 0]]
# using BoW - if a word appear in text nothing to do with frequency

# Label encoder assigns nos to keys in vocab

# processed_docs = ["dog bites man","man bites dog","dog and dog are friends"]

# Our vocabulary:  {'and': 0,'are': 1,'bites': 2,'dog': 3,'friends': 4,'man': 5}
# BoW representation for 'dog bites man':  [[0 0 1 1 0 1]]
# BoW representation for 'man bites dog:  [[0 0 1 1 0 1]]
# Bow representation for 'dog and dog are friends': [[1 1 0 2 1 0]]



# CountVectorizer(ngram_range=(1,3))
#Our vocabulary:  [('bites', 0), ('bites dog', 1), ('bites man', 2), 
# ('dog', 3), ('dog bites', 4), ('dog bites man', 5), ('dog eats', 6), 
# ('dog eats , 7), ('eats', 8), ('eats food', 9), ('eats meat', 10), 
# ('food', 11), ('man', 12), ('man bites', 13), meat'('man bites dog', 14), 
# ('man eats', 15), ('man eats food', 16), ('meat', 17)]


# BoW representation for 'dog bites man':  [[1 0 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0]]
# BoW representation for 'man bites dog:  [[1 1 0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0]]
# Bow representation for 'dog and dog are friends': [[0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

# Note that the number of features (and hence the size of the feature vector) 
# increased a lot for the same data, compared to the ther single word based 
# representations!!





# Clipping each gradient matrix individually changes their relative scale but 
# is also possible,Despite what seems to be popular, you probably want to clip 
# the whole gradient by its global norm:

# optimizer = tf.train.AdamOptimizer(1e-3)
# gradients, variables = zip(*optimizer.compute_gradients(loss))
# # with tf.GradientTape() as tape:
# #   loss = ...
# # variables = ...
# # gradients = tape.gradient(loss, variables)
# gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
# # gradients = [
# #     None if gradient is None else tf.clip_by_norm(gradient, 5.0)
# #     for gradient in gradients]
# optimize = optimizer.apply_gradients(zip(gradients, variables))



# model = tf.keras.models.Sequential([...])
# model.compile(optimizer=tf.keras.optimizers.SGD(clipvalue=0.5),loss,metrics)
# model.fit(train_data,steps_per_epoch,epochs)

### Tensorflow

### PyTorch

Tf-Idf 
-------
The meaning increases proportionally to the number of times in the 
text a word appears but is compensated by the word frequency in 
the corpus (data-set)

tf(t,d) = count of t in d / number of words in d
df(t) = occurrence of t in documents

df(t) = N(t)
where
df(t) = Document frequency of a term t
N(t) = Number of documents containing the term t
idf(t) = N/ df(t) = N/N(t)


#for strange word embedding representation for full text in spacy gives null vector
temp = nlp('practicalnlp is a newword')
temp[0].vector

#skip-gram else its 0 for CBOW. Default is CBOW.

from gensim.models import Word2Vec, KeyedVectors 

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize




### Parsing Url
from urllib.parse import urlparse
import re

# http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html

# x =  train.loc[0:0,"url"].apply(urlparse)
# x[0]
# ParseResult(scheme='http', netloc='www.bloomberg.com', 
# path='/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html', 
# params='', query='', fragment='')


# train.loc[:,"website"] = train.loc[:,"url"].apply(urlparse).apply(
# 	lambda x:x[1].replace("www.","").replace(".com",""))

# train.loc[:,"year"] = train.loc[:,"url"].apply(urlparse).apply(
#     lambda x:"/".join(x[2:])).apply(
#     lambda x:re.findall(r"\d{4}", x)).apply(
#         lambda x: int(x[0]) if len(x) > 0 else np.nan).apply(
#             lambda x: x if x>1990 and x<2050 else np.nan )
        
# yearList = list(train["year"].unique())
# map_dic = {}
# for year in yearList:
#     if pd.isnull(year) != True:
#         map_dic[str(int(year))] = "Year"
# train.loc[:,"website_type"] = train.loc[:,"url"].apply(urlparse).apply(
#     lambda x:x[2].split("/")[1]).apply(
#         lambda x: map_dic[x] if x in map_dic.keys() else x)
    

# train.loc[:,"website_sub_type"] = train.loc[:,"url"].apply(urlparse).apply(
#     lambda x: x[2].split("/")[2] if ((len(x) > 2) and len(x[2].split("/"))>2) else "?").apply(
#         lambda x: map_dic[x] if x in map_dic.keys() else x)
    

# train.loc[:,"domain"] = train.loc[:,"url"].apply(urlparse).apply(
# 	lambda x: x[1].split(".")[-1])


df2 = pd.DataFrame()
df2["website_name"] = train.loc[:,"url"].apply(urlparse).apply(
	lambda x:x[1].replace("www","").split(".")[-2])
df2["website_domain"] = train.loc[:,"url"].apply(urlparse).apply(
	lambda x:x[1].split(".")[-1])

df2["website_type"] = train.loc[:,"url"].apply(urlparse).apply(
	lambda x:x[2].split("/")[1])

df2["website_content"] = train.loc[:,"url"].apply(urlparse).apply(
    lambda x:"".join(x[2].split("/")[2:]) 
    if(len(x[2].split("/"))>2) else x[2].split("/")[-1])

df2["website_date"] = train.loc[:,"url"].apply(urlparse).apply(
	lambda x: re.findall(r"\d{4}",x[2]))
df2["website_date"]=df2["website_date"].apply(
	lambda x:np.nan if (len(x)==0) else x[0])

df2["website_content"]=df2["website_content"].apply(
	lambda x:re.sub(r"\d{4}-\d{2}|\d*","",x)) # remove date & beginning with digits
df2["website_content"]=df2["website_content"].apply(
	lambda x:re.sub(r"\d|\.(.*)","",x)) #remove digits & .html,.php etc

==================================================================================
# Home Depot Product Search Relevance
# Predict the relevance of search results on homedepot.com
# Home Depot is asking Kagglers to help them improve their customers' 
# shopping experience by developing a model that can accurately predict the 
# relevance of search results.
==================================================================================
## Fixing typos in search term
----------------------------------
START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;'),
)

def spell_check(s):
    q = '+'.join(s.split())
    time.sleep(randint(0,2) ) #relax and don't let google be angry
    r = requests.get("https://www.google.co.uk/search?q="+q)
    content = r.text
    start=content.find(START_SPELL_CHECK) 
    if ( start > -1 ):
        start = start + len(START_SPELL_CHECK)
        end=content.find(END_SPELL_CHECK)
        search= content[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in HTML_Codes:
            search = search.replace(code[1], code[0])
        search = search[1:]
    else:
        search = s
    return search ;


##samples
searches = [ "metal plate cover gcfi", "artric air portable", "roll roofing lap cemet",
             "basemetnt window","vynal grip strip", "lawn mower- electic" ]
 
for search in searches:
    speel_check_search= spell_check(search)
    print (search+"->" + speel_check_search)

# metal plate cover gcfi->metal plate cover gcfi
# artric air portable->artric air portable
# roll roofing lap cemet->roll roofing lap cemet
# basemetnt window->basemetnt window
# vynal grip strip->vynal grip strip
# lawn mower- electic->lawn mower- electic


'''
Recommendation system part I: Product pupularity based system targetted at 
new customers

# get most popular product -> gr by ProductId + rating count + plot top 30

Recommendation system part II: Model-based collaborative filtering system 
based on customer's purchase history and ratings provided by other users 
who bought items similar items

Model-based collaborative filtering system
-----------------------------------------------
Recommend items to users based on purchase history and similarity of ratings 
provided by other users who bought items to that of a particular customer.
A model based collaborative filtering technique is closen here as it helps in 
making predictinfg products for a particular user by identifying patterns based 
on preferences from multiple user data.

dataset :
        
    UserId          ProductId   Rating  Timestamp
0   A39HTATAQ9V7YF  0205616461  5.0     1369699200
1   A3JM6GV9MNOF9X  0558925278  3.0     1355443200
2   A1Z513UWSAAO0F  0558925278  5.0     1404691200
3   A1WMRR494NWEWV  0733001998  4.0     1382572800
4   A3IAAVS479H7M7  0737104473  1.0     1274227200


Recommendation system part III: When a business is setting up its e-commerce 
website for the first time withou any product rating

For a business without any user-item purchase history, 
a search engine based recommendation system can be designed for users. 
The product recommendations can be based on textual clustering analysis given 
in product description.

dataset:
    product_uid     product_description
0   100001          Not only do angles make joints stronger, they ...
1   100002          BEHR Premium Textured DECKOVER is an innovativ...
2   100003          Classic architecture meets contemporary design...
3   100004          The Grape Solar 265-Watt Polycrystalline PV So...
4   100005          Update your bathroom with the Delta Vero Singl...

'''


# Recommendation System - Part II
-------------------------------------


# Utility Matrix based on products sold and user reviews
---------------------------------------------------------------------------------
# Utility Matrix : An utlity matrix is consists of all possible 
# user-item preferences (ratings) details represented as a matrix. 
# The utility matrix is sparce as none of the users would buy all teh items 
# in the list, hence, most of the values are unknown.

X = amazon_ratings1.pivot_table(values='Rating', index='UserId', 
            columns='ProductId', fill_value=0).T

# ProductId  X  UserId  => (886,9697)                                                                            

# Now Decomposing the Matrix (using SVD with n_components=10) # (886, 10) 
#            + correlation_matrix = np.corrcoef(decomposed_matrix)

i=X.index[99] #'6117036094'
product_names = list(X.index)
product_ID = product_names.index(i) #99
correlation_product_ID = correlation_matrix[product_ID] #(886,)
Recommend = list(X.index[correlation_product_ID > 0.90])
# Removes the item already bought by the customer
Recommend.remove(i) 

# Recommendation System - Part III
-------------------------------------
# use vectorizer + use init=k-means++ 
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
# Top terms per cluster:
# Cluster 0:
#  concrete  # terms[idx] for idx in order_centroids[i, :10]
#  stake
#  ft
#  coating
#  apply
#  epoxy

def show_recommendations(product):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])

show_recommendations("cutting tool")
# Cluster 4:
#  cutting
#  saw
#  tool
#  blade
#  non

================================================================================
# Avito Duplicate Ads Detection - Can you detect duplicitous duplicate ads?
#Develop a model that can automatically spot duplicate ads. 
=============================================================================

# Avito Duplicate Ads Detection
-------------------------------------------------------------------
### Category.csv.zip -> (categoryID,parentCategoryID)
### Images_0 ~ Images_9.zip
### ItemInfo_test.csv.zip -> same as ItemInfo_train
### ItemInfo_train.csv.zip -> (itemID,categoryID,title,attrsJSON,locationID,
#                                     images_array,price,description,metroID,lat,lon)
### ItemPairs_test.csv.zip  -> (id,itemID_1,itemID_2)
### ItemPairs_train.csv.zip -> (itemID_1,itemID_2,isDuplicate:{0,1},generationMethod:{1,2,3})
### Location.csv.zip -> (locationID,regionID)

# types = {feature_name : dtype }
pairs = pd.read_csv("../input/ItemPairs_train.csv", dtype=types1)
# Add 'id' column for easy merge
items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)
items,location,category 
train = pairs
# add length for title,description,attrsJSON in items 
# left join items1(subset with limited feature) with category on='categoryID'  
# & location on='locationID'
# left join train with items1 on='itemID_1'
# do the same for items2 with train on='itemID_2'

# create same array for every required features of train
train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)

# features = get_features(list of columns except 'itemID_1' & 'itemID_2') 
# call XGB (train,test,features,'isDuplicate')

