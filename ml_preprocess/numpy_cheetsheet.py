np.ravel()
np.c_[]
pixel_means = X_train.mean(axis=0, keepdims=True)
np.squeeze()
np.take()
np.meshgrid()
np.expand_dims()


x0, x1 = np.meshgrid(
    np.linspace(0, 5, 5).reshape(-1, 1), 
    np.linspace(0, 2, 2).reshape(-1, 1), 
)
# x0 : 
# [[0.   1.25 2.5  3.75 5.  ]
#  [0.   1.25 2.5  3.75 5.  ]]

# x1: 
# [[0. 0. 0. 0. 0.]
#  [2. 2. 2. 2. 2.]]
X_new = np.c_[x0.ravel(), x1.ravel()]
# X_new :
# [[0.  , 0.  ],
# [1.25, 0.  ],
# [2.5 , 0.  ],
# [3.75, 0.  ],
# [5.  , 0.  ],
# [0.  , 2.  ],
# [1.25, 2.  ],
# [2.5 , 2.  ],
# [3.75, 2.  ],
# [5.  , 2.  ]]

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_train[index])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
    
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data['comment_text'] = data['comment_text'].apply(removeStopWords)