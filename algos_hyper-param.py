SVM (Support Vector Machine)
===================================
Pros
--------------------
1. Performs well in Higher dimension. In real world there are infinite 
dimensions (and not just 2D and 3D). For instance image data, gene data, 
medical data etc. has higher dimensions and SVM is useful in that. 
Basically when the number of features/columns are higher, SVM does well

2. Best algorithm when classes are separable. (when instances of both 
the classes can be easily separated either by a straight line or 
non-linearly). To depict separable classes, lets take an example(here 
taking an example of linear separation, classes can also be non-linearly 
separable, by drawing a parabola for e.g. etc). In first graph you cannot 
tell easily whether X will be in class 1 or 2, but in case 2 you can 
easily tell that X is in class 2. Hence in second case classes are 
linearly separable.

First is non-separable class, second is separable class.

3. Outliers have less impact.

4. SVM is suited for extreme case binary classification.

Cons:
-------------
1. Slow: For larger dataset, it requires a large amount of time to process.

2. Poor performance with Overlapped classes : Does not perform well 
in case of overlapped classes.

3. Selecting appropriate hyperparameters is important: That will allow 
for sufficient generalization performance.

4. Selecting the appropriate kernel function can be tricky.

Applications:
------------------------
Bag of words application(many features and columns), speech recognition 
data, classification of images(non-linear data), medical 
analytics(non linear data), text classification(many features)

=============================================================================

Naive Bayes
=================

Pros
-----------------
1. Real time predictions: It is very fast and can be used in real time.

2. Scalable with Large datasets

3. Insensitive to irrelevant features.

4. Multi class prediction is effectively done in Naive Bayes

5. Good performance with high dimensional data(no. of features is large)

Cons
------------------
1. Independence of features does not hold: The fundamental Naive Bayes 
assumption is that each feature makes an independent and equal contribution 
to the outcome. However this condition is not met most of the times.

2. Bad estimator: Probability outputs from predict_proba are not to be 
taken too seriously.

3. Training data should represent population well: If you have no 
occurrences of a class label and a certain attribute value together 
(e.g. class=”No”, shape=”Overcast “) then the posterior probability will 
be zero. So if the training data is not representative of the population, 
Naive bayes does not work well.(This problem is removed by smoothening 
techniques).

Applications:
-------------------------------
Naive Bayes is used in Text classification/ Spam Filtering/ Sentiment 
Analysis. It is used in text classification (it can predict on multiple 
classes and doesn’t mind dealing with irrelevant features), Spam filtering 
(identify spam e-mail) and Sentiment Analysis (in social media analysis, 
to identify positive and negative sentiments), recommendation systems 
(what will the user buy next)

=============================================================================

Logistic Regression
========================
Pros
--------------
1. Simple to implement

2. Effective

3. Feature scaling not needed: Does not require input features to be 
scaled (can work with scaled features too, but doesn’t require scaling)

3. Tuning of hyperparameters not needed.

Cons
--------------------
1. Poor performance on non-linear data(image data for e.g)

2. Poor performance with irrelevant and highly correlated features 
(use Boruta plot for removing similar or correlated features and 
    irrelevant features).

3. Not very powerful algorithm and can be easily outperformed by other 
algorithms.

4. High reliance on proper presentation of data. All the important 
variables / features should be identified for it to work well.

Applications:
--------------------------
Any classification problem that is preferably binary (it can also 
perform multi class classification, but binary is preferred). For example 
you can use it if your output class has 2 outcomes; cancer detection 
problems, whether a student will pass/fail, default/no default in case of 
customer taking loan, whether a customer will churn or not, email is spam 
or not etc.

=============================================================================

Random Forest
===================
Pros:
--------------
1. Random forest can decorrelate trees. It picks the training sample 
and gives each tree a subset of the features(suppose training data 
was [1,2,3,4,5,6], so one tree will get subset of training 
data [1,2,3,2,6,6]. Note that size of training data remains same, both 
datas have length 6 and that feature ‘2’ and feature ‘6’ are repeated in 
the randomly sampled training data given to one tree. Each tree predicts 
according to the features it has. In this case tree 1 only has access to 
features 1,2,3 and 6 so it can predict based on these features. Some other 
tree will have access to features 1,4,5 say so it will predict according 
to those features. If features are highly correlated then that problem can 
be tackled in random forest.

2. Reduced error: Random forest is an ensemble of decision trees. 
For predicting the outcome of a particular row, random forest takes 
inputs from all the trees and then predicts the outcome. This ensures 
that the individual errors of trees are minimized and overall variance 
and error is reduced.

3. Good Performance on Imbalanced datasets : It can also handle errors 
in imbalanced data (one class is majority and other class is minority)

4. Handling of huge amount of data: It can handle huge amount of data 
with higher dimensionality of variables.

5. Good handling of missing data: It can handle missing data very well. 
So if there is large amount of missing data in your model, it will give 
good results.

6. Little impact of outliers: As the final outcome is taken by consulting 
many decision trees so certain data points which are outliers will not 
have a very big impact on Random Forest.

7. No problem of overfitting: In Random forest considers only a subset of 
features, and the final outcome depends on all the trees. So there is 
more generalization and less overfitting.

8. Useful to extract feature importance (we can use it for feature selection)

Cons:
----------------------
1. Features need to have some predictive power else they won’t work.

2. Predictions of the trees need to be uncorrelated.

3. Appears as Black Box: It is tough to know what is happening. 
You can at best try different parameters and random seeds to change 
the outcomes and performance.

Applications:
-------------------------
Credit card default, fraud customer/not, easy to identify patient’s disease or not, recommendation system for ecommerce sites.

===========================================================================
Decision Trees
=====================
Pros
-----------
1. Normalization or scaling of data not needed.

2. Handling missing values: No considerable impact of missing values.

3. Easy to explain to non-technical team members.

4. Easy visualization

5. Automatic Feature selection : Irrelevant features won’t affect 
decision trees.

Cons
-----------
1. Prone to overfitting.

2. Sensitive to data. If data changes slightly, the outcomes can change to a 
very large extent.

3. Higher time required to train decision trees.

4.For data including categorical variables with different numbers of levels,
 information gain in decision trees is biased in favor of attributes 
 with more levels. However, the issue of biased predictor selection is 
 avoided by the Conditional Inference approach, a two-stage approach, 
 or adaptive leave-one-out feature selection.

Applications:
------------------------------
Identifying buyers for products, prediction of likelihood of default, 
which strategy can maximize profit, finding strategy for cost minimization, 
which features are most important to attract and retain customers 
(is it the frequency of shopping, is it the frequent discounts, is it 
the product mix etc), fault diagnosis in machines(keep measuring pressure, 
vibrations and other measures and predict before a fault occurs) etc.
==========================================================================

XGBoost
===================
Pros
-------------
1. Less feature engineering required (No need for scaling, 
normalizing data, can also handle missing values well)

2. Feature importance can be found out(it output importance of each 
feature, can be used for feature selection)

3. Fast to interpret

4. Outliers have minimal impact.

5. Handles large sized datasets well.

6. Good Execution speed

7. Good model performance 

8. Less prone to overfitting

Cons
---------------
1. Difficult interpretation , visualization tough

2. Overfitting possible if parameters not tuned properly.

3. Harder to tune as there are too many hyperparameters.

Applications
----------------------
Any classification problem. Specially useful if you have too many 
features and too large datasets, outliers are present, there are many 
missing values and you don’t want to do much feature engineering. 

When your data is made of heterogeneous columns such as age, weight, 
number of times the client called, average time of a call, etc then 
Xgboost is usually better than Deep Learning.


k-NN (K Nearest Neighbors)
===================================
Pros
---------
1. Simple to understand and impelment

2. No assumption about data (for e.g. in case of linear regression we 
assume dependent variable and independent variables are linearly related, 
in Naïve Bayes we assume features are independent of each other etc., 
but k-NN makes no assumptions about data)

3. Constantly evolving model: When it is exposed to new data, it changes 
to accommodate the new data points.

4. Multi-class problems can also be solved.

5. One Hyper Parameter: K-NN might take some time while selecting the 
first hyper parameter but after that rest of the parameters are aligned to it.

Cons
-------------------
1. Slow for large datasets.

2. Curse of dimensionality: Does not work very well on datasets with 
large number of features.

3. Scaling of data absolute must.

4. Does not work well on Imbalanced data. So before using k-NN either 
undersamplemajority class or oversample minority class and have a 
balanced dataset.

5. Sensitive to outliers.

6. Can’t deal well with missing values

Applications:
========================

You can use it for any classification problem when dataset is smaller, 
and has lesser number of features so that computation time taken by 
k-NN is less. If you do not know the shape of the data and the way output 
and inputs are related (whether classes can be separated by a line or 
ellipse or parabola etc.), then you can use k-NN.

===========================================================================

Computational Complexities of different ML Models.
============================================================
Assumptions:
n = number of training examples, m = number of features, 
n'' = number of support vectors,
k = number of neighbors, k'' = number of trees

    Linear Regression
        Train Time Complexity=O(n*m^2 + m^3)
        Test Time Complexity=O(m)
        Space Complexity = O(m)

    Logistic Regression
        Train Time Complexity=O(n*m)
        Test Time Complexity=O(m)
        Space Complexity = O(m)

    K Nearest Neighbors
        Train Time Complexity=O(k*n*m)
        Test Time Complexity=O(n*m)
        Space Complexity = O(n*m)

    SVM
        Train Time Complexity=O(n^2)
        Test Time Complexity=O(n''*m)
        Space Complexity = O(n*m)

    Decision Tree
        Train Time Complexity=O(n*log(n)*m)
        Test Time Complexity=O(m)
        Space Complexity = O(depth of tree)

    Random Forest # Bagging
        Train Time Complexity=O(k''*n*log(n)*m)
        Test Time Complexity=O(m*k'')
        Space Complexity = O(k''*depth of tree)

    Naive Bayes
        Training Time Complexity = O(n*m)
        Test Time Complexity=O(m)
        Run-time Complexity = O(c*m)

    Gradient boosted decision tree(GBDT): # GB,adaboost,xgboost
        Training Time Complexity = O(k''*n*logn*m)
        Train space complexity = O(nodes_count(k'')+leaf_values(k''))
        Run-time Complexity = O(k'' logn)
        Run-time Space Complexity = O(nodes_count(k'')*k''+leaf_values(k''))



Important HyperParameters
============================
Logistic Regression
============================
# solver
-------------
'lbfgs' -> performs well & save memory, but may face issue with convergence
'sag' -> fast for large data
'saga' -> for sparse multinomial log reg. & very large data , 
            need to use feature scaling(pref. minmax)
'newton-cg' -> computationally expensive due to Hessian mat.
'liblinear' -> high dimensional data,need to use regularization along

#penalty/regularization
-----------------------------
'Penalty' -> {'l1','l2','elasticnet'}
# some penalty doesn't work with some solver

# regularization strength
-----------------------------
'C' -> Smaller values specify stronger regularization and high value tells 
the model to give high weight to the training data.

============================================================================
Polynomial Regression
=============================
interaction_only : (bool, default=False)
--------------------------------------------
# If True, only interaction features are produced: features that are products 
# of at most degree distinct input features, i.e. terms with power of 2 or 
# higher of the same input feature are excluded:

# included: x[0], x[1], x[0] * x[1], etc.

# excluded: x[0] ** 2, x[0] ** 2 * x[1], etc.

# X
# array([[0, 1],
#        [2, 3],
#        [4, 5]])
## with interaction_only=False , degree=2
# array([[ 1.,  0.,  1.,  0.,  0.,  1.],
#        [ 1.,  2.,  3.,  4.,  6.,  9.],
#        [ 1.,  4.,  5., 16., 20., 25.]])
poly = PolynomialFeatures(interaction_only=True)
poly.fit_transform(X)
# array([[ 1.,  0.,  1.,  0.],
#        [ 1.,  2.,  3.,  6.],
#        [ 1.,  4.,  5., 20.]])
============================================================================

KNN
=====
kneighbors_graph(X=None, n_neighbors=None, mode='connectivity'):
 Compute the (weighted) graph of Neighbors for points in X.

X = [[0], [3], [1]] # create object then fit()
A = neigh.kneighbors_graph(X) 

metric : str or callable, default=’minkowski’,can use scipy.spatial.distance
for other valid metric , 
p : int, default=2 (euclidean_distance) -> For arbitrary p, minkowski_distance (l_p) is used.
----------------------------------------------------

weights : {‘uniform’, ‘distance’} or callable, default=’uniform’
-------------------------------------------------------------------
‘uniform’ : uniform weights. All points in each neighborhood are weighted 
equally.

‘distance’ : weight points by the inverse of their distance. in this case, 
closer neighbors of a query point will have a greater influence than 
neighbors which are further away.

algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
---------------------------------------------------------------------
    Algorithm used to compute the nearest neighbors:

        ‘ball_tree’ will use BallTree

        ‘kd_tree’ will use KDTree

        ‘brute’ will use a brute-force search.

        ‘auto’ will attempt to decide the most appropriate algorithm based 
        on the values passed to fit method.

    Note: fitting on sparse input will override the setting of this 
    parameter, using brute force.
=============================================================================

SVM (classifier)
==================
kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, 
default=’rbf’ : 
------------------------------------------------------------------------
If a callable is given it is used to pre-compute the kernel matrix from 
data matrices; that matrix should be an array of shape (n_samples, n_samples).

probability : bool, default=False
--------------------------------------
    Whether to enable probability estimates. This must be enabled prior 
    to calling fit, will slow down that method as it internally uses 
    5-fold cross-validation, and predict_proba may be inconsistent 
    with predict.

tol : float, default=1e-3
----------------------------
    Tolerance for stopping criterion.

cache_size : float, default=200
-----------------------------------
    Specify the size of the kernel cache (in MB).

     
decision_function_shape : {‘ovo’, ‘ovr’}, default=’ovr’
----------------------------------------------------------
C : float, default=1.0 -> Regularization parameter.Must be strictly 
positive. The penalty is a squared l2 penalty

=============================================================================

Decision Tree
===============
criterion : {“gini”, “entropy”, “log_loss”}, default=”gini”
-----------------------------------------------------------------
The function to measure the quality of a split. Supported criteria are 
“gini” for the Gini impurity and “log_loss” and “entropy” both for 
the Shannon information gain

max_depth : int, default=None
---------------------------------

min_samples_split : int or float, default=2
---------------------------------------------
The minimum number of samples required to split an internal node:



If int, then consider min_samples_split as the minimum number.

If float, then min_samples_split is a fraction and 
ceil(min_samples_split * n_samples) are the minimum number of samples 
for each split.


min_samples_leaf : int or float, default=1
----------------------------------------------------
The minimum number of samples required to be at a leaf node. 
A split point at any depth will only be considered if it leaves at 
least min_samples_leaf training samples in each of the left and 
right branches. This may have the effect of smoothing the model, 
especially in regression.

max_features : int, float or {“auto”, “sqrt”, “log2”}, default=None
----------------------------------------------------------------------
The number of features to consider when looking for the best split:
If “log2”, then max_features=log2(n_features).

min_impurity_decrease : float, default=0.0
------------------------------------------------------
A node will be split if this split induces a decrease of the impurity 
greater than or equal to this value.

The weighted impurity decrease equation is the following:

N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)

where N is the total number of samples, N_t is 
the number of samples at the current node, N_t_L is the number of 
samples in the left child, and N_t_R is the number of samples in the 
right child.

N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight 
is passed.

ccp_alpha : non-negative float, default=0.0
----------------------------------------------
Complexity parameter used for Minimal Cost-Complexity Pruning. 
The subtree with the largest cost complexity that is smaller than 
ccp_alpha will be chosen. By default, no pruning is performed.

max_leaf_nodesint, default=None
----------------------------------
Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are 
defined as relative reduction in impurity. If None then unlimited number 
of leaf nodes


=============================================================================

Naive Bays (MultinomialNB)
=============================
alpha :float or array-like of shape (n_features,), default=1.0
----------------------------------------------------------------
Additive (Laplace/Lidstone) smoothing parameter 
(set alpha=0 and force_alpha=True, for no smoothing).

fit_prior : bool, default=True
---------------------------------------
Whether to learn class prior probabilities or not. If false, a uniform 
prior will be used.

class_prior : array-like of shape (n_classes,), default=None
---------------------------------------------------------------
Prior probabilities of the classes. If specified, the priors are 
not adjusted according to the data.


=======================================================================

Random Forest
=====================
# all decision tree parameters

bootstrap : bool, default=True
-----------------------------------
Whether bootstrap samples are used when building trees. If False, 
the whole dataset is used to build each tree.

oob_score : bool, default=False
-------------------------------------
Whether to use out-of-bag samples to estimate the generalization 
score. Only available if bootstrap=True.

warm_start : bool, default=False
--------------------------------------
When set to True, reuse the solution of the previous call to fit and 
add more estimators to the ensemble, otherwise, just fit a whole new 
forest. 
