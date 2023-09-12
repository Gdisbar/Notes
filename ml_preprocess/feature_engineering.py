# Feature Selection Tool for Machine Learning 
=============================================================
# check out : https://www.featuretools.com/


1. missing value -> lower = 40%, upper = 60 %

2. correlated feature -> threshold = 95% -> use pairplot,crosstab,correlation matrix
# Create correlation matrix
corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features 
df.drop(to_drop, axis=1, inplace=True)

3. find feature importance -> add some threshold & remove (like take top 10)

4. Variance Threshold -> only for Unsupervised Learning 


A feature with a higher variance means that the value within that feature 
varies or has a high cardinality. On the other hand, lower variance means the 
value within the feature is similar, and zero variance means you have a 
feature with the same value.

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
threshold=1
selector = VarianceThreshold(threshold)
selector.fit(df)
df.columns[selector.get_support()]

### For supervised learning

5. Univariate feature selection (chi2, Pearson-correlation)


# For regression: r_regression, f_regression, mutual_info_regression
# For classification: chi2, f_classif, mutual_info_classif

#Select top 2 features based on mutual info regression
selector = SelectKBest(mutual_info_regression, k =2) 
selector.fit(X, y)
X.columns[selector.get_support()]

6. Using SelectFromModel -> SelectFromModel feature selection is based on 
the importance attribute (often is coef_ or feature_importances_ but 
it could be any callable) , similar to RFE but not same

# #Selecting the Best important features according to 
# Logistic Regression using SelectFromModel
sfm_selector = SelectFromModel(estimator=LogisticRegression())
sfm_selector.fit(X, y)
X.columns[sfm_selector.get_support()]


7. Sequential Feature Selection -> greedy based on cross-validation score 

sfs_selector = SequentialFeatureSelector(estimator=LogisticRegression(), 
			n_features_to_select = 3, cv =10, direction ='backward')
sfs_selector.fit(X, y)
X.columns[sfs_selector.get_support()]

===================================================================================
### Multivariate Feature selection
====================================================================================
1. crate feature combination -> using feature transformation
2. use ElasticNet
poly = PolynomialFeatures(2) # (x1,x2) -> (1,x1,x2,x1*x1,x1*x2,x2*x2)
poly = PolynomialFeatures(degree=3, interaction_only=True)
#(x1,x2,x3) -> (1,x1,x2,x3,x1*x2,x1*x3,x2*x3,x1*x2*x3)
poly.fit_transform(X)
spline = SplineTransformer(degree=2, n_knots=3)
# array([[0],
#        [1],
#        [2],
#        [3],
#        [4]])
# array([[0.5  , 0.5  , 0.   , 0.   ],
#        [0.125, 0.75 , 0.125, 0.   ],
#        [0.   , 0.5  , 0.5  , 0.   ],
#        [0.   , 0.125, 0.75 , 0.125],
#        [0.   , 0.   , 0.5  , 0.5  ]])

# Only the three middle diagonals are non-zero for degree=2. 
# The higher the degree, the more overlapping of the splines.

# Interestingly, a SplineTransformer of degree=0 is the same as 
# KBinsDiscretizer with encode='onehot-dense' and n_bins = n_knots - 1 
# if knots = strategy.

transformer = FunctionTransformer(np.log1p, validate=True)
transformer.transform(X)
# X = np.array([[0, 1], [2, 3]])
# array([[0.        , 0.69314718],
#        [1.09861229, 1.38629436]])



## As a part of pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)


==============================================================================
|| When to do feature scaling ?   
==============================================================================
-------------------------
| Tree Based Algorithm  |  
--------------------------
Algorithms that rely on rules so no effect of monotonic transformation
CART, Random Forests, Gradient Boosted Decision Trees,
Linear Discriminant Analysis(LDA), Naive Bayes is by design equipped to 
handle this and give weights to the features accordingly. 

Few key points to note :

    Mean centering does not affect the covariance matrix
    Scaling of variables does affect the covariance matrix
    Standardizing affects the covariance


--------------------------------------
| Gradient Descent Based Algorithms |
--------------------------------------
Having features on a similar scale will help the gradient descent converge 
more quickly towards the minima.

Specially for NN -> 

    It makes the training faster
    It prevents the optimization from getting stuck in local optima
    It gives a better error surface shape
    Weight decay and Bayes optimization can be done more conveniently

------------------------------
| Distance-Based Algorithms |
------------------------------
KNN, K-means, and SVM  use distances between data points to determine 
their similarity and hence perform the task at hand. Therefore, 
we scale our data before employing a distance-based algorithm 
so that all the features contribute equally to the result.

--------------------------------------
| For feature engineering using PCA |
-------------------------------------
In PCA we are interested in the components that maximize the 
variance. If one component (e.g. age) varies less than another (e.g. salary) 
because of their respective scales, PCA might determine that the direction 
of maximal variance much more important 

---------------
| Regression |
--------------
When one variable has a very large scale,the regression coefficients may 
be on a very small -> normalization will work

While creating power terms, If you don’t center X first, your squared term 
will be highly correlated with X, which could muddy the estimation of the 
coefficients. 

reating interaction terms: If an interaction/product term is created from 
two variables that are not centered on 0, some amount of collinearity 
will be induced 


Centering/scaling does not affect your statistical inference in 
regression models — the estimates are adjusted appropriately and 
the p-values will be the same. The scale and location of the explanatory 
variables do not affect the validity of the regression model in any way.

============================
| Types of Feature scaling |
============================
from sklearn.preprocessing  # library

use fit() + transform() on X_train  & only trasform() on X_test data

-----------------
| # MinMaxScaler  -> 
-----------------

x_new = x - min(x)/max(x)-min(x) 

This Scaler shrinks the data within the range of -1 to 1 if there are 
negative values. We can set the range like [0,1] or [0,5] or [-1,1].

This Scaler responds well if the standard deviation is small and when a 
distribution is not Gaussian. This Scaler is sensitive to outliers.

x_new = x - min(x)*(b-a)/max(x)-min(x) # output range in [a,b]

--------------------
| # StandardScaler ->
---------------------

x_new = x - mean(x)/std(x)

The Standard Scaler assumes data is normally distributed within each 
feature and scales them such that the distribution centered around 0, 
with a standard deviation of 1.

Centering and scaling happen independently on each feature by computing 
the relevant statistics on the samples in the training set. 
If data is not normally distributed, this is not the best Scaler to use.

----------------
| # MaxAbsScaler
------------------

maximal absolute value of each feature in the training set is 1.0. 
It does not shift/center the data and thus does not destroy any sparsity.

On positive-only data, this Scaler behaves similarly to Min Max 
Scaler and, therefore, also suffers from the presence of significant outliers.

-------------------
| # Robust Scaler
-------------------

If our data contains many outliers, scaling using the mean and standard 
deviation of the data won’t work well. This Scaler removes the median 
and scales the data according to the quantile range IQR

The centering and scaling statistics of this Scaler are based on 
percentiles and are therefore not influenced by a few numbers of 
huge marginal outliers. Note that the outliers themselves are still 
present in the transformed data. If a separate 
outlier clipping is desirable, a non-linear transformation is required.

------------------------------------------------
| # Quantile Transformer Scaler / Rank scaler
-------------------------------------------------

This method transforms the features to follow a uniform or a normal 
distribution. Therefore, for a given feature, this transformation tends 
to spread out the most frequent values. It also reduces the impact of 
(marginal) outliers

The cumulative distribution function of a feature is used to 
project the original values. Note that this transform is non-linear 
and may distort linear correlations between variables measured at the 
same scale but renders variables measured at different scales more 
directly comparable. 

It  is useful when we have a large dataset with many data points 
usually more than 1000.

-----------------------------
| # Power Transformer Scaler
-----------------------------

The power transformer is a family of parametric, monotonic transformations 
that are applied to make data more Gaussian-like. This is useful for 
modeling issues related to the variability of a variable that is unequal 
across the range (heteroscedasticity) or situations where normality is desired.

The power transform finds the optimal scaling factor in stabilizing 
variance and minimizing skewness through maximum likelihood estimation. 

# It uses Box-Cox requires input data to be strictly positive, 
# while Yeo-Johnson supports both positive or negative data.

------------------------
| # Unit Vector Scaler
------------------------

 x_new = x/norm(x)

Scaling is done considering the whole feature vector to be of unit length. 
This usually means dividing each component by the Euclidean length 
of the vector (L2 Norm). In some applications (e.g., histogram features), 
it can be more practical to use the L1 norm of the feature vector.

Like Min-Max Scaling, the Unit Vector technique produces values of 
range [0,1]. When dealing with features with hard boundaries, this is 
quite useful. For example, 
when dealing with image data, the colors can range from only 0 to 255.

df1 = df.apply(lambda x : x/np.linalg.norm(x,1)) # L1 norm

===================================================================================================
================================================================================
############## Model with low number of samples ##############################
================================================================================

# model will overfit

from sklearn.ensemble import AdaBoostRegressor
import numpy as np

np.random.seed(42)
y_train = np.random.rand(55)
X_train = np.random.rand(55, 7500)

model = AdaBoostRegressor(random_state=i)
model.fit(X_train, y_train)
model.score(X_train, y_train)
## 0.9895214625949762

# clear overfit , don't agree
# check out the iter no to generate the same R2 score
# for predicting just 10 test features . If you wait long enough, you would find 
# a completely random solution, on completely random data, that matches your test 
# data well. The same applies to data-based feature selection. What we're 
# seeing is the 
# underlying algorithm of random number generator

np.random.seed(42)

best_r2 = 0
y_test = np.random.rand(10)

for i in range(10000):
    y_pred = np.random.rand(10)
    r, _ = sp.pearsonr(y_pred, y_test)
    r2 = r**2
    if r2 > best_r2:
        best_r2 = r2
        print(f"iter={i}, r={r2}")

## iter=0, r=0.49601681572673695
## iter=6, r=0.6467516405878888
## iter=92, r=0.6910478084107202
## iter=458, r=0.6971821688682832
## iter=580, r=0.6988719722383485
## iter=1257, r=0.721148489188462
## iter=2015, r=0.7437673627048644
## iter=2253, r=0.7842495052355497
## iter=4579, r=0.8189207386492211
## iter=5465, r=0.8749525244481782

If your sample size is already small I recommend avoiding any 
data driven optimization. Instead, restrict yourself to models where you can 
fix hyperparameters by your knowledge about model and application/data. 
This makes one of the validation/test levels unnecessary, leaving more of your 
few cases for training of the surrogate models in the remaining cross 
validation.


The more iterations you make, the more likely you are to overfit. 
Avoid data-based optimization as much as possible. So rather than 
using data-based feature selection, pick (< 10) meaningful features by hand.


Same applies to model choice, with such small data, you don''t want to 
tune the hyperparameters. Use domain knowledge to pick a model that is 
likely to work for the data. You want a simple model that is less likely 
to overfit.


Use cross-validation rather than held-out set. The test set consisting 
of ten samples is too small and unreliable. You could easily overfit to it.


================================================================================
############## Model with high number of features ##############################
================================================================================
1. Univariate/Multivariate feature selection using statistical test
2. Recursive Feature Elimination - Use Feature importance using SHAP value 
            or feature importance for tree base algo

# Recursive Feature Elimination
-------------------------------------
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

selector = RFE(estimator, n_features_to_select=5, step=1)
#selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.support_
# array([ True,  True,  True,  True,  True, False, False, False, False,
#        False])
selector.ranking_
# array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

support_ -> gives the result (based on the selected model and no of requirement) 
with respect to features sequentially.


ranking_[i] -> corresponds to the ranking position of the i-th feature. 
Selected (i.e., estimated best) features are assigned rank 1.

# Recursive Feature Elimination 
-----------------------------------
from yellowbrick.model_selection import RFECV
visualizer = RFECV(RandomForestClassifier(), cv=cv, scoring='f1_weighted')
visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

3. using L1 regularization for linerar & logistic regression that makes weak 
coefficients to 0 , other way to do it is to use 
Least Angle Regression + L1,Coordinate Descent , SGD+warm_start 
use BoLasso -> L1 + (combine with L2,stability selection) to tackle instability 
in Lasso which usees Lasso Bootstrapping

4. Dimensionality Reduction
5. Nnet based techniques like Autoencoders