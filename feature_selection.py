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