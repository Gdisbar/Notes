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