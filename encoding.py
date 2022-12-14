# Different types of Encoding 
==============================
# Converting categorical features to numeric ones

Categorical variables:['Suburb', 'Address', 'Type', 'Method', 'SellerG', 
'Date', 'CouncilArea', 'Regionname']
features = df[['Type','Method','Regionname']]

# Handling Categorical Variables : Replace using the map function
mapping = {'h':1,'u':2,'t':3}
features['type'] = features.Type.map(mapping,na_action='ignore') 

# difference between apply(), map(), applymap()
===================================================
# For applying a function on 1D arrays to each column or row. 
# DataFrame’s apply method does exactly this:

# f = lambda x: x.max() - x.min()
# df.apply(f)


# Many of the most common array statistics (like sum and mean) are DataFrame 
# methods, so using apply is not necessary.

# Element-wise Python functions can be used, too. Suppose you wanted to 
# compute a formatted string from each floating point value in frame. 
# You can do this with applymap:

# f = lambda x: '%.2f' % x
# df.applymap(f)

# The reason for the name applymap is that Series has a map method for 
# applying an element-wise function:

# Summing up, apply works on a row / column basis of a DataFrame, 
# applymap works element-wise on a DataFrame, and map works element-wise 
# on a Series.



p=pd.Series([1,2,3])
# 0    1
# 1    2
# 2    3
# dtype: int64

p.apply(lambda x: pd.Series([x, x]))
#    0  1
# 0  1  1
# 1  2  2
# 2  3  3

p.map(lambda x: pd.Series([x, x]))
# 0    0    1
# 1    1
# dtype: int64
# 1    0    2
# 1    2
# dtype: int64
# 2    0    3
# 1    3
# dtype: int64
# dtype: object

# DEFINITION
--------------
# map is defined on Series ONLY
# applymap is defined on DataFrames ONLY
# apply is defined on BOTH

# INPUT ARGUMENT
------------------
# map accepts dicts, Series, or callable
# applymap and apply accept callables only

# BEHAVIOR
--------------
# map is elementwise for Series
# applymap is elementwise for DataFrames
# apply also works elementwise but is suited to more complex 
# operations and aggregation. The behaviour and return value depends on 
# the function.

# USE CASE
-------------------------------------------------------------
# map is meant for mapping values from one domain to another, so is 
# optimised for performance (e.g., df['A'].map({1:'a', 2:'b', 3:'c'}))

# applymap is good for elementwise transformations across multiple 
# rows/columns (e.g., df[['A', 'B', 'C']].applymap(str.strip))

# apply is for applying any function that cannot be vectorised 
# (e.g., df['sentences'].apply(nltk.sent_tokenize)).


# df

#      Name  Value                       Title
# 0  mickey     20                  wonderland
# 1  donald     10  welcome to donald's castle
# 2  minnie     86      Minnie mouse clubhouse

mask = df.apply(lambda x: x['Name'].lower() in x['Title'].lower(), axis=1)
df[mask]

#      Name                       Title  Value
# 1  donald  welcome to donald's castle     10
# 2  minnie      Minnie mouse clubhouse     86

# https://stackoverflow.com/questions/54432583/when-should-i-not-want-to-use-pandas-apply-in-my-code


# 1. Nominal categorical variable
====================================
# variables for which we do not have to worry about the arrangement of the 
# categories.Ex. Gender, State

#Types of Nominal Encoding
----------------------------
#    1. One-Hot Encoding 
#    2. One hot Encoding (Multiple Categories) — Nominal Categories

# 1.1 one hot encoding
===================================================================
# One hot vectors make the number of columns = number of categories
df2 = pd.get_dummies(features['Method'])

#Dummy Variable Trap
=======================================================================
# if we have n columns, then the one hot encoding should create n-1 columns.

# For encoding Gender(male=1,female=0) 

# If we keep both newly formed columns (Male, Female) in our data, 
# it causes Multicollinearity that affects machine learning algorithm. 
# Multicollinearity occurs due to fact that if a particular row have 
# Gender as Male, Male column have value 1 and Female column will obviously 
# have 0 value. It is called dummy variable trap. Relation between Male and 
# Female column is:

# Value in Male Column = 1- Value in Female Column

# To avoid multicollinearity we drop one of the column (either Male or Female)

# If we talk about Multicollinearity, it does not affect the model 
# predictive accuracy, but it undermines the statistical significance of the 
# independent variables. 

# To remove multicollinearity, first we need to find the correlation 
# between all the independent variables (features) of our dataset. 

# Correlation between GarageCars and GarageArea is 0.88, that is quite high. 
# So, to remove multicollinearity you can drop any one column 
# (either GarageCars or GarageArea).

# Create dummies without the baseline
------------------------------------------
dum_trap=pd.get_dummies(data=df_cat)

# There are a total of 43 dummy variables created but the problem with 
# these variables is multicollinearity. Because there is no base value 
# for any of these. Let’s set a baseline for every dummy variable by using 
# the function drop_first.

# Creating dummies with baseline
--------------------------------------
dum=pd.get_dummies(data=df_cat,drop_first=True)


# The total number of dummy variables created is reduced to 26 from 43 and 
# the problem of multicollinearity is also reduced.

# 1.2 One hot Encoding (Multiple Categories) — Nominal Categories 
================================================================================
df = sns.load_dataset('titanic')
data = df['sex']
print(data.head(5))
# 0      male
# 1    female
# 2    female
# 3    female
# 4      male
# Name: sex, dtype: object
print(pd.concat([data,pd.get_dummies(data)],axis=1).head())
#       sex  female  male
# 0    male       0     1
# 1  female       1     0
# 2  female       1     0
# 3  female       1     0
# 4    male       0     1
data=df['embarked']
print(data.head(5))
# 0    S
# 1    C
# 2    S
# 3    S
# 4    S
# Name: embarked, dtype: object
print("number of unique feature : ",data.unique())
# number of unique feature :  ['S' 'C' 'Q' nan]
print("set of dummy variable\n",pd.get_dummies(data).head())
# set of dummy variable
#    C  Q  S
# 0  0  0  1
# 1  1  0  0
# 2  0  0  1
# 3  0  0  1
# 4  0  0  1
print("get k-1 dummy variable\n",pd.get_dummies(data,drop_first=True).head())
# get k-1 dummy variable
#    Q  S
# 0  0  1
# 1  0  0
# 2  0  1
# 3  0  1
# 4  0  1
print("add additional dummy for nan\n",pd.get_dummies(data,drop_first=True,dummy_na=True).head())
# add additional dummy for nan
#    Q  S  NaN
# 0  0  1    0
# 1  0  0    0
# 2  0  1    0
# 3  0  1    0
# 4  0  1    0
print("count each dummy variable name\n",pd.get_dummies(data,drop_first=True,dummy_na=True).sum(axis=0))
# count each dummy variable name
# Q       77
# S      644
# NaN      2
# dtype: int64


# when to use k & k-1 dummy variable ?

# k-1 features are used for regression,svm,neural net,clustering
-------------------------------------------------------------------------
# One hot encoding into k-1 binary variables takes into account so that 
# we can use 1 less dimension and still represent the whole information: if 
# the observation is 0 in all the binary variables, then it must be 1 in the 
# final (removed) binary variable. This is valid for all algorithms that look at 
# ALL the features at the same time during training. 
# (By keeping the degree of freedom to k-1)
 

# k features are used for (tree based model,feature selection)
------------------------------------------------------------------------------
# For tree based models select at each iteration only a group of features 
# to make a decision. So if we use k-1 features then 1 feature willbe left out 
# during splits.So the dropped feature will never be considered
 

# 2. Ordinal Categorical variable 
===================================================
# variables for which we have to worry about the rank. These categories can be 
# rearranged based on ranks. Ex. education (PHD-1, masters-2, bachelors-3)


from sklearn.preprocessing import OrdinalEncoder
import numpy as np
enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
print("Encoder fit : ",enc.fit(X))
# Encoder fit :  OrdinalEncoder()
print("Encoder categories : ",enc.categories_)
# Encoder categories :  [array(['Female', 'Male'], dtype=object), 
# 					array([1, 2, 3], dtype=object)]
print("Encoder transform : \n",enc.transform([['Female', 3], ['Male', 1]]))
# Encoder transform : 
#  [[0. 2.]
#  [1. 0.]]
print("Encoder Inverse Transform : \n",enc.inverse_transform([[1, 0], [0, 1]]))
# Encoder Inverse Transform : 
#  [['Male' 1]
#  ['Female' 2]]
X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
print("Encoder fit_transform :\n",enc.fit_transform(X))
# Encoder fit_transform :
#  [[ 1.  0.]
#  [ 0.  1.]
#  [ 0. nan]]
print("Encoder with missing value treatment :\n",
	enc.set_params(encoded_missing_value=-1).fit_transform(X))
# Encoder with missing value treatment
#  [[ 1.  0.]
#  [ 0.  1.]
#  [ 0. -1.]]

# Types of Ordinal Encoding
-----------------------------
# 	1. Label Encoding
#   2. Label Binarizer
#   3. Count/Frequency Encoder

# 2.1.Label Encoding
=======================================================
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = features[['Regionname']]
df1['Region'] = le.fit_transform(features['Regionname'])
df1.value_counts()


# 2.2. Label Binarizer
====================================
from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(features["Type"])
pd.DataFrame(lb_results, columns=lb_style.classes_).value_counts()


# comparing OneHotEncoder,LabelEncoder,LabelBinarizer
=============================================================================
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 
'warm', 'hot']
values = array(data)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# Label Encoder: [0 0 2 0 1 1 2 0 2 1]

# onehot encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# OneHot Encoder:
#  [[1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]

#Binary encode
lb = LabelBinarizer()
lb.fit_transform(values)

# Label Binarizer:
#  [[1 0 0]
#  [1 0 0]
#  [0 0 1]
#  [1 0 0]
#  [0 1 0]
#  [0 1 0]
#  [0 0 1]
#  [1 0 0]
#  [0 0 1]
#  [0 1 0]]

# A difference is that you can use OneHotEncoder for multi column data, 
# while not for LabelBinarizer and LabelEncoder.
------------------------------------------------------------------

X = [["US", "M"], ["UK", "M"], ["FR", "F"]]
OneHotEncoder().fit_transform(X).toarray()

# array([[0., 0., 1., 0., 1.],
#        [0., 1., 0., 0., 1.],
#        [1., 0., 0., 1., 0.]])

LabelBinarizer().fit_transform(X)
# ValueError: Multioutput target data is not supported with label binarization

LabelEncoder().fit_transform(X)
# ValueError: bad input shape (3, 2)


# 2.3 Count/Frequency Encoder
============================================================================
mapping = features.Type.value_counts().to_dict()
features.Type = features.Type.map(mapping,na_action='ignore')


# 	id 	bin_0 	bin_1 	bin_2 	bin_3 	bin_4 	nom_0 	nom_1 	nom_2 	nom_3 	... 	nom_9 	ord_0 	ord_1 	ord_2 	ord_3 	ord_4 	ord_5 	day 	month 	target
# 0 	0 	0 	0 	0 	T 	Y 	Green 	Triangle 	Snake 	Finland 	... 	2f4cb3d51 	2 	Grandmaster 	Cold 	h 	D 	kr 	2 	2 	0
# 1 	1 	0 	1 	0 	T 	Y 	Green 	Trapezoid 	Hamster 	Russia 	... 	f83c56c21 	1 	Grandmaster 	Hot 	a 	A 	bF 	7 	8 	0
# 2 	2 	0 	0 	0 	F 	Y 	Blue 	Trapezoid 	Lion 	Russia 	... 	ae6800dd0 	1 	Expert 	Lava Hot 	h 	R 	Jc 	7 	2 	0
# 3 	3 	0 	1 	0 	F 	Y 	Red 	Trapezoid 	Snake 	Canada 	... 	8270f0d71 	1 	Grandmaster 	Boiling Hot 	i 	D 	kW 	2 	1 	1
# 4 	4 	0 	0 	0 	F 	N 	Red 	Trapezoid 	Lion 	Canada 	... 	b164b72a7 	1 	Grandmaster 	Freezing 	a 	R 	qP 	7 	8 	0

# 5 rows × 25 columns

# Frequency of each unique categorical values of feature "nom_1"
enc_nom_1 = (data.groupby('nom_1').size()) / len(data)

# nom_1
# Circle       0.124400
# Polygon      0.120477
# Square       0.165323
# Star         0.153013
# Trapezoid    0.337270
# Triangle     0.099517
# dtype: float64

data['nom_1_encode'] = data['nom_1'].apply(lambda x : enc_nom_1[x])


# 	id 	bin_0 	bin_1 	bin_2 	bin_3 	bin_4 	nom_0 	nom_1 	nom_2 	nom_3 	... 	ord_0 	ord_1 	ord_2 	ord_3 	ord_4 	ord_5 	day 	month 	target 	nom_1_encode
# 0 	0 	0 	0 	0 	T 	Y 	Green 	Triangle 	Snake 	Finland 	... 	2 	Grandmaster 	Cold 	h 	D 	kr 	2 	2 	0 	0.099517
# 1 	1 	0 	1 	0 	T 	Y 	Green 	Trapezoid 	Hamster 	Russia 	... 	1 	Grandmaster 	Hot 	a 	A 	bF 	7 	8 	0 	0.337270
# 2 	2 	0 	0 	0 	F 	Y 	Blue 	Trapezoid 	Lion 	Russia 	... 	1 	Expert 	Lava Hot 	h 	R 	Jc 	7 	2 	0 	0.337270
# 3 	3 	0 	1 	0 	F 	Y 	Red 	Trapezoid 	Snake 	Canada 	... 	1 	Grandmaster 	Boiling Hot 	i 	D 	kW 	2 	1 	1 	0.337270
# 4 	4 	0 	0 	0 	F 	N 	Red 	Trapezoid 	Lion 	Canada 	... 	1 	Grandmaster 	Freezing 	a 	R 	qP 	7 	8 	0 	0.337270

# 5 rows × 26 columns

# Target Encoding
===========================================================================
# Target encoding is the method of converting a categorical value into 
# the mean of the target variable. This type of encoding is a type of 
# bayesian encoding method where bayesian encoders use target variables 
# to encode the categorical value.

# The target encoding encoder calculates the mean of the target variable 
# for each category and by the mean, the categories get replaced. 


# This is a good method for encoding: using this we can encode any number 
# of categories. But it can cause the overfitting of any model because 
# we are using mean as a category and this generates a hard correlation 
# between these features.

# Using this we can train the model but in testing, it can lead to the 
# failure or inaccuracy of the model.


# When you have many categories, good to use target encoding over one-hot
# Target encoding cannot capture dependencies between different categorical 
# features

# Example: Salary | city=Seattle, college=Seattle Central College > 
#   Salary | city=Salt Lake or SFO, college= Seattle Central College


# Target-mean encoding: we replace the category with the mean of the target 
# values. This method will usually be used with smoothing to avoid target leakage.


data = [['Salt Lake City', 10, 120], ['Seattle', 5, 120], ['San Franscisco', 5, 140], 
        ['Seattle', 3, 100], ['Seattle', 1, 70], ['San Franscisco', 2, 100],['Salt Lake City', 1, 60], 
        ['San Franscisco', 2, 110], ['Seattle', 4, 100],['Salt Lake City', 2, 70] ]
df = pd.DataFrame(data, columns = ['City', 'Years OF Exp','Yearly Salary in Thousands'])

#              City  Years OF Exp  Yearly Salary in Thousands
# 0  Salt Lake City            10                         120
# 1         Seattle             5                         120
# 2  San Franscisco             5                         140
# 3         Seattle             3                         100
# 4         Seattle             1                          70
# 5  San Franscisco             2                         100
# 6  Salt Lake City             1                          60
# 7  San Franscisco             2                         110
# 8         Seattle             4                         100
# 9  Salt Lake City             2                          70
tenc=ce.TargetEncoder(smoothing=1) 
df_city=tenc.fit_transform(df['City'],df['Yearly Salary in Thousands'])
df_new = df_city.join(df.drop('City',axis = 1))

#          City  Years OF Exp  Yearly Salary in Thousands
# 0   85.200846            10                         120
# 1   97.571139             5                         120
# 2  114.560748             5                         140
# 3   97.571139             3                         100
# 4   97.571139             1                          70
# 5  114.560748             2                         100
# 6   85.200846             1                          60
# 7  114.560748             2                         110
# 8   97.571139             4                         100
# 9   85.200846             2                          70

# Leave-one-out encoding: this method is very similar to target mean encoding, 
# but the difference is that in leave-one-out encoding, we take the mean of 
# the target values of all the samples except the one we want to predict.

tenc=ce.LeaveOneOutEncoder() 
#          City  Years OF Exp  Yearly Salary in Thousands
# 0   65.000000            10                         120
# 1   90.000000             5                         120
# 2  105.000000             5                         140
# 3   96.666667             3                         100
# 4  106.666667             1                          70
# 5  125.000000             2                         100
# 6   95.000000             1                          60
# 7  120.000000             2                         110
# 8   96.666667             4                         100
# 9   90.000000             2                          70

# Effect Encoding / Sum Encoding / Deviation
===================================================
# In this type of encoding, encoders provide values to the categories in -1,0,1 
# format. -1 formation is the only difference between One-Hot encoding and 
# effect encoding.

data=df['embarked']
data.iloc[4]=np.nan # previously it was S
print(data.head(6))
# 0      S
# 1      C
# 2      S
# 3      S
# 4    NaN
# 5      Q
# Name: embarked, dtype: object
encoder=ce.sum_coding.SumEncoder(cols='embarked',verbose=True)
data=encoder.fit_transform(data)
print(data.head(6))
#    intercept  embarked_0  embarked_1  embarked_2
# 0          1         1.0         0.0         0.0
# 1          1         0.0         1.0         0.0
# 2          1         1.0         0.0         0.0
# 3          1         1.0         0.0         0.0
# 4          1         0.0         0.0         1.0
# 5          1        -1.0        -1.0        -1.0


# Hash Encoder / Feature Hasing
===================================
# Hashing is a one-way technique of encoding which is unlike other encoders. 
# The Hash encoder’s output can not be converted again into the input. 
# That is why we can say it may cause loss of information from the data. 
# It should be applied with high dimension data in terms of categorical values.

tenc=ce.HashingEncoder() 
df_city=tenc.fit_transform(df['City']) 
# print(df['City'].unique()) # ['Salt Lake City' 'Seattle' 'San Franscisco']
df_new = df_city.join(df.drop('City',axis = 1))
print(df_new)
#    col_0  col_1  col_2  ...  col_7  Years OF Exp  Yearly Salary in Thousands
# 0      0      1      0  ...      0            10                         120
# 1      0      0      0  ...      1             5                         120
# 2      0      0      0  ...      0             5                         140
# 3      0      0      0  ...      1             3                         100
# 4      0      0      0  ...      1             1                          70
# 5      0      0      0  ...      0             2                         100
# 6      0      1      0  ...      0             1                          60
# 7      0      0      0  ...      0             2                         110
# 8      0      0      0  ...      1             4                         100
# 9      0      1      0  ...      0             2                          70

# [10 rows x 10 columns]

tenc=ce.HashingEncoder(n_components=3)
#    col_0  col_1  col_2  Years OF Exp  Yearly Salary in Thousands
# 0      0      1      0            10                         120
# 1      0      1      0             5                         120
# 2      0      0      1             5                         140
# 3      0      1      0             3                         100
# 4      0      1      0             1                          70
# 5      0      0      1             2                         100
# 6      0      1      0             1                          60
# 7      0      0      1             2                         110
# 8      0      1      0             4                         100
# 9      0      1      0             2                          70

# Binarey Encoding
================================================================================
# using hashing can cause the loss of data and on the other hand we have seen 
# in one hot encoding dimensionality of the data is increasing. The binary 
# encoding is a process where we can perform hash encoding look like encoding 
# without losing the information just like one hot encoding.

data=pd.DataFrame({'Month':['January','April','March','April','Februay',
	'June','July','June','September']}) 
# ['January' 'April' 'March' 'Februay' 'June' 'July' 'September']
#        Month
# 0    January
# 1      April
# 2      March
# 3      April
# 4    Februay
# 5       June
# 6       July
# 7       June
# 8  September
encoder= ce.BinaryEncoder(cols=['Month'],return_df=True)
data=encoder.fit_transform(data) 
#    Month_0  Month_1  Month_2
# 0        0        0        1
# 1        0        1        0
# 2        0        1        1
# 3        0        1        0
# 4        1        0        0
# 5        1        0        1
# 6        1        1        0
# 7        1        0        1
# 8        1        1        1


# Base-N Encoding
================================================================================
# In base n encoding if the base is two then the encoder will convert 
# categories into the numerical form using their respective binary form 
# which is formally one-hot encoding. But if we change the base to 10 which 
# means the categories will get converted into numeric form between 0-9. 

# comparing with Binary Encoder , base=5
-------------------------------------------
# binary encoding = 4 dimensions , binary n-encoding = 3 dimensions 
# In the above output, we can see that we have used base 5. Somewhere it is 
# pretty simple to the binary encoding but where in binary we have got 4 
# dimensions after conversion here we have 3 dimensions only and also the 
# numbers are varying between 0-4. 

# If we do not define the base by default it is set to 2 which basically 
# performs the binary encoding.

# ['January' 'April' 'March' 'Februay' 'June' 'July' 'September']
encoder= ce.BaseNEncoder(cols=['Month'],return_df=True,base=5)
data=encoder.fit_transform(data)
print(data)
#    Month_0  Month_1
# 0        0        1
# 1        0        2
# 2        0        3
# 3        0        2
# 4        0        4
# 5        1        0
# 6        1        1
# 7        1        0
# 8        1        2



# So, which one should you use?
====================================================================================
# most commonly used = one hot encoding for nominal variables
# avoid curse of dimensionality = binary encoding (alternative to one-hot)

# for ordinal variables = use ordinal encoding (easily reversible & doesn''t affect
# 	dimensionality)
# need to capture independent & dependent feature + supervised + 
# 				doesn''t increase affect dimensionality = target encoding (but be 
# 								careful about overfitting & target leakage) 


# supervised + doesn''t increase affect dimensionality = frequency & count
# 				encoder (can only be used when count refer to target variable,
# 				otherwise, all categories that have similar cardinality will 
# 				be counted the same.)
