=============================
Polynomial Regression
=============================

# numpy.polyfit: Used to fit a polynomial function of given degrees. It returns the coefficients (which minimizes the squared error) of the polynomial equation

#     This function takes three arguments to fit a polynomial function to given data. X values, Y values and degrees (1,2,3...n)
#     Polynomial equation of degree 1 is linear equation: y = mx + b
#     Polynomial equation of degree 2 is quadratic equation: y = ax^2 + bx + c
#     So polyfit function will return the coefficients of the polynomial equation. For linear equation coeff will be [m,b] and for quadratic equation coeff will be [a,b,c]

# numpy.poly1d: polynomial function using the coefficients returned by 'numpy.polyfit'

# It takes polynomial coefficients as argument and construct a polynomial.
# For e.g. if there are three coeff then it will construct quadratic equation

# numpy.linspace: Takes three arguments (start, stop and num) and generates evenly spaced values(same as 'num') within 'start' and 'stop' range
# Finaly to draw a polynomial function we will plot 'values' generated 'linspace' function on X axis and on Y axis 'poly_func' output for every 'values'

polynomial_plot(X_train.Width, y_train) # here Width is the feature name

def polynomial_plot(feature, label):
  
    x_coordinates = feature
    y_coordinates = np.squeeze(label)

    # Contruct first degree polynomial function

    linear_func = np.poly1d(np.polyfit(x_coordinates, y_coordinates, 1))
    # Contruct second degree polynomial function
    quadratic_func = np.poly1d(np.polyfit(x_coordinates, y_coordinates, 2))
 
    # Generate evenly spaced values
    values = np.linspace(x_coordinates.min(), x_coordinates.max(), len(x_coordinates))

    plt.scatter(x_coordinates,y_coordinates, color='blue')  
    plt.plot(values, linear_func(values), color='cyan', linestyle='dashed', label='Linear Function')
    plt.plot(values, quadratic_func(values), color='red', label='Quadratic Function')
    plt.xlabel('%s From Test Data'%(feature.name))
    plt.ylabel('Weight')
    plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
    plt.legend()
    plt.title("Linear Vs Quadratic Function For Feature %s" % (feature.name))
    plt.show()  


================================================================================

# pip install mlxtend
from mlxtend.evaluate import bias_variance_decomp
mse, bias, var = bias_variance_decomp(model, X_train, y_train, 
    X_test, y_test, loss='mse', num_rounds=200, random_seed=1)

# manual calculation
--------------------------
def get_bias(predicted_values, true_values):
    return np.round(np.mean((predicted_values - true_values) ** 2), 0)

def get_variance(values):
    return np.round(np.var(values), 0)


def get_metrics(y_train, y_test, train_pred, test_pred):
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    bias = get_bias(test_predictions, y_test)
    var = get_variance(test_pred)
    
    return [train_mse,test_mse,bias,var]

degree=3
poly = np.polyfit(x_train, y_train, deg=degree)
p_3rd = np.poly1d(poly.reshape(1, degree+1)[0]) 
train_pred = np.polyval(poly,x_train)
test_pred =  np.polyval(poly,x_test)
train_mse,test_mse,bias,variance = get_metrics(y_train, y_test,train_pred,
                                                        test_pred)


# data augmentation
==========================================================================
x = [1,2,3,4]
x10 = np.arange(min(x),max(x),0.1) # add 10 data points in between x
# array([1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2,
#        2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5,
#        3.6, 3.7, 3.8, 3.9])

# Monte-Carlo simulation / SMOTE
==================================
# Before using a pseudorandom sequence in a Monte Carlo Simulation, 
# a thorough analysis of the randomness of the sequence should be carried out. 
# First, you must test for uniformity by applying the chi-square and the 
# Kolmogorov-Smirnov tests. Serial correlation (same as autocorrelation) 
# could be tested with the Durbin-Watson test. Every random sequence must be 
# tested for runs, for example using the Wald-Wolfowitz test. 
# You must also test for digit patterns using the gap test and/or the poker test.




# compare classification algorithm
--------------------------------------------------------

y_scores = cross_val_predict(estimator=sgd_clf, X=X_train, y=y_train_5, 
                             cv=3, method='decision_function', n_jobs=-1)
# getting best threshold for precision & recalll balance

precision, recall, thresholds = precision_recall_curve(
    y_true=y_train_5, probas_pred=y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.grid()

plot_precision_recall_vs_threshold(precision, recall, thresholds)
plt.show()

# what is the recall value for 90% precision

threshold_90_precision = thresholds[np.argmax(precision > .9)]
threshold_90_precision

# make predictions for the training set, you first get the score of 
# the desired point and compare the score to the chosen threshold:

y_train_pred_90 = (y_scores > threshold_90_precision)

# check these predictions' precision and recall:
precision = precision_score(y_true=y_train_5, y_pred=y_train_pred_90)

recall = recall_score(y_train_5, y_train_pred_90)

#roc curve 
---------------------------------
fpr, tpr, thresholds = roc_curve(y_true=y_train_5, y_score=y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
plot_roc_curve(fpr, tpr, label='SGD')
plt.legend(loc='lower right')
plt.show()

# roc curve - SGD ~ random-forest
-----------------------------------------
y_probas_forest = cross_val_predict(estimator=forest_clf, X=X_train, 
                                    y=y_train_5, cv=5, 
                                    n_jobs=-1, method='predict_proba')

y_scores_forest = y_probas_forest[:, 1]  # P(X=5)


fpr_forest, tps_forest, threshs_forest = roc_curve(
    y_true=y_train_5, y_score=y_scores_forest)

plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tps_forest, label="Random Forest")
plt.legend(loc="lower right")
plt.show()


# Error analysis - classification
====================================
conf_mtrx = confusion_matrix(y_true=y_train, y_pred=y_train_predict)
# array([[2806,    0,   15,    4,    4,   29,   17,    2,   84,    0],
#        [   1, 3249,   26,   14,    3,   24,    3,    3,   93,    7],
#        [  14,   17, 2587,   57,   38,   14,   38,   21,  156,    6],
#        [  13,   15,   72, 2651,    2,   95,   11,   27,  150,   37],
#        [   9,    8,   27,    7, 2623,    4,   20,   10,  120,   98],
#        [  16,   11,   19,   82,   39, 2253,   37,    7,  204,   41],
#        [  16,    6,   22,    0,   22,   46, 2801,    5,   57,    0],
#        [  11,    3,   36,   16,   28,    2,    3, 2821,   53,  134],
#        [   6,   35,   23,   58,    3,   55,   23,    3, 2648,   21],
#        [  16,   13,   17,   31,   61,   15,    1,   86,  123, 2640]])

plt.matshow(A=conf_mtrx, cmap=plt.cm.gray)
plt.show()
# Instead of plotting the absolute numbers, we will plot counts 
# over the number of images of the class:

row_sums = conf_mtrx.sum(axis=1, keepdims=True)
normed_conf_mtrx = conf_mtrx / row_sums
# Then we fill the diagonal with zeros to highlight only the errors
np.fill_diagonal(normed_conf_mtrx, val=0)
plt.matshow(A=normed_conf_mtrx, cmap=plt.cm.gray)
plt.show()

# check misclassified class 3 & 5
===========================================

cl_a, cl_b = 3, 5 # these 2 was showing most errors

X_aa = X_train[(y_train == cl_a) & (y_train_predict == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_predict == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_predict == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_predict == cl_b)]

def plot_digits(instances, images_per_row=10, **options):
    """Plots digits on a grid of rows and columns
    
    # Arguments
        instances: np.ndarray, the digits, where each is a flat array
        images_per_row: int, how many digits to be displayed per row
        options: other arguments for `plt.imshow()`
    """
    size = 28
    n_images = instances.shape[0]
    images_per_row = min(images_per_row, n_images)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = ceil(n_images / images_per_row)
    row_images = list()
    n_empty = (n_rows * images_per_row) - n_images
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row*images_per_row : (row+1)*images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap='binary', **options)
    plt.axis('off')

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

===========================================================================
Financial Time series 
===========================================================================
# Removing noise with the Fourier Transform
--------------------------------------------------
# Fourier filters the noise at different levels of 
# n_components (above which the coefficients will be kept)
# The bigger the value the more frequencies we remove. 
# The trick here is to find a value that keeps the trend but removes 
# most of the noise.

def fft_denoiser(x, n_components, keep_complex=False): 

    n = len(x)
    fft = np.fft.fft(x, n)
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    clean_data = np.fft.ifft(fft)
    if not keep_complex:
        clean_data = clean_data.real
    
    return clean_data

# financial time series are considered non-stationary 
# (although any attempt to prove it statistically is doomed), 
# thus making Fourier a bad choice.
# The Kalman Filter is essentially a Bayesian Linear Regression that 
# can optimally estimate the hidden state of a process using its observable 
# variables.

# Dealing with Outliers
-------------------------
def basic_filter(data, mode='rolling', window=262, threshold=3):
    
    msg = f"Type must be of pandas.Series but {type(data)} was passed."
    assert isinstance(data, pd.Series), msg
    
    series = data.copy()
    
    # rolling/expanding objects
    pd_object = getattr(series, mode)(window=window)
    mean = pd_object.mean()
    std = pd_object.std()
    
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    outliers = ~series.between(lower_bound, upper_bound)
    # fill false positives with 0
    outliers.iloc[:window] = np.zeros(shape=window)
    
    series = series.to_frame()
    series['outliers'] = np.array(outliers.astype('int').values)
    series.columns = ['Close', 'Outliers']
    
    return series


# normalize time series data
--------------------------------
from sklearn.base import BaseEstimator, TransformerMixin


class RollingStandardScaler(BaseEstimator, TransformerMixin):
    """Rolling standard Scaler
    
    Standardized the given data series using the mean and std 
    commputed in rolling or expanding mode.
    
    Parameters
    ----------
    window : int
        Number of periods to compute the mean and std.
    mode : str, optional, default: 'rolling'
        Mode 
        
    Attributes
    ----------
    pd_object : pandas.Rolling
        Pandas window object.
    w_mean : pandas.Series
        Series of mean values.
    w_std : pandas.Series
        Series of std. values.
    """
    def __init__(self, window, mode='rolling'):
        self.window = window
        self.mode = mode
        
        # to fill in code
        self.pd_object = None
        self.w_mean = None
        self.w_std = None
        self.__fitted__ = False
        
    def __repr__(self):
        return f"RollingStandardScaler(window={self.window}, mode={self.mode})"
        
    def fit(self, X, y=None):
        """Fits.
        
        Computes the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_shape, n_features)
            The data used to compute the per-feature mean and std. Used for
            later scaling along the feature axis.
        y
            Ignored.
        """
        self.pd_object = getattr(X, self.mode)(self.window)
        self.w_mean = self.pd_object.mean()
        self.w_std = self.pd_object.std()
        self.__fitted__ = True
        
        return self
    
    def transform(self, X):
        """Transforms.
        
        Scale features of X according to the window mean and standard 
        deviation.
        
        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.
        
        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed data.
        """
        self._check_fitted()
        
        standardized = X.copy()
        return (standardized - self.w_mean) / self.w_std
    
    def inverse_transform(self, X):
        """Inverse transform
        
        Undo the transform operation
        
        Paramaters
        ----------
        X : array-like of shape (n_shape, n_features)
            Input data that will be transformed.
        
        Returns
        -------
        standardized : array-like of shape (n_shape, n_features)
            Transformed (original) data.
        """
        self._check_fitted()
        
        unstandardized = X.copy()
        return  (unstandardized * self.w_std) + self.w_mean
        
    def _check_fitted(self):
        """ Checks if the algorithm is fitted. """
        if not self.__fitted__:
            raise ValueError("Please, fit the algorithm first.")


# compute returns
-------------------
def compute_returns(data, periods=1, log=False, relative=True):
    """Computes returns.
    Calculates the returns of a given dataframe for the given period. The 
    returns can be computed as log returns or as arithmetic returns
    
    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The data to calculate returns of.
    periods : int
        The period difference to compute returns.
    log : bool, optional, default: False
        Whether to compute log returns (True) or not (False).
    relative : bool, optional, default: True
        Whether to compute relative returns (True) or not 
        (False).
    
    Returns
    -------
    ret : pandas.DataFrame or pandas.Series
        The computed returns.
    
    """
    if log:
        if not relative:
            raise ValueError("Log returns are relative by definition.")
        else:
            ret = _log_returns(data, periods)
    else:
        ret = _arithmetic_returns(data, periods, relative)

    return ret


def _arithmetic_returns(data, periods, relative):
    """Arithmetic returns."""
    # to avoid computing it twice
    shifted = data.shift(periods)
    ret = (data - shifted) 
    
    if relative:
        return ret / shifted
    else:
        return ret


def _log_returns(data, periods):
    """Log returns."""
    return np.log(data / data.shift(periods))



===========================================================================
## Outlier removal
==========================================================================
1. calculate outlier using IQR
2. group features & check which one has outlier
3. remove outlier which is common accross all groups contrary to removing 
	all outlier
=========================================================================
## Data cleaning
## Encoding 
# squeeze -> function to reduce the 2D array to 1D array
# ravel -> returns contiguous flattened array(1D array with all the input-array elements and with the same type as it
# c_ -> 

======================
# m>>n
=====================
Projection Methods -> SVD,PCA
Recursive Feature elemination -> remove highly correlated features
Use regularization

For uncorrelated features, the optimal feature size is N−1 
(where N is sample size)

As feature correlation increases, and the optimal feature size becomes 
proportional to sqrt(N) for highly correlated features.

Another (empirical) approach that could be taken, is to draw the 
learning curves for different sample sizes from the same dataset, 
and use that to predict classifier performance at different sample sizes

Use following Algos
---------------------
 SIFT, HOG, Shape context, SSIM, GM and 4 DNN-based features.

# How to detect and deal with Multicollinearity
===================================================
# 3D plotting + plane
======================
x_line = df["X1"].values.reshape(-1,1)
y_line = df["X2"].values.reshape(-1,1)
z_line = df["X3"].values.reshape(-1,1)
     
X = np.hstack((x, y))
X = np.hstack((np.ones((x.shape[0], 1)), X ))
theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), z)
k = int(max(np.max(x), np.max(y), np.max(z))) #size of the plane

p1, p2 = np.mgrid[:k, :k]
P = np.hstack(( np.reshape(p1, (k*k, 1)), np.reshape(p2, (k*k, 1))))
P = np.hstack(( np.ones((k*k, 1)), P))

plane = np.reshape(np.dot(P, theta), (k, k));

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(x[:, 0], y[:,0], z[:,0],'ro') # scatter plot
ax.plot_surface(p1,p2,plane) # plane plot

ax.set_xlabel('x1 label')
ax.set_ylabel('x2 label')
ax.set_zlabel('y label')
plt.show()

---------------------------------------------------------------------
Correlation Technique| Relation   | Datatype of features
-------------------------------------------------------------------------
Pearson              | Linear     | Quantitative & Quantitative
------------------------------------------------------------------------
Spearman             | Non-Linear | Ordinal & Ordinal
-------------------------------------------------------------------------
Point-biserial       | Linear     | Binary & Quantitative
-------------------------------------------------------------------------
Cramer''s V          | Non-Linear | Categorical & Categorical
-------------------------------------------------------------------------
Kendall''s tau       | Non-Linear | Two Categorical & Two Quantitative
------------------------------------------------------------------------- 

VIF(Variation Inflation Factor) is used to identify the correlation of one 
independent variable with a group of other variables.

VIF = 1 → No correlation
VIF = 1 to 5 → Moderate correlation
VIF >10 → High correlation


def vif_scores(df):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independent Features"] = df.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values,i) 
                                    for i in range(df.shape[1])]
    return VIF_Scoresdf1 = df.iloc[:,:-1]
vif_scores(df1)

# Fixing Multicollinearity 
===============================================
Combining variables

Dropping variables

Lasso Regression - increasing the alpha value for the L1 regularizer, 
we introduce some small bias in the estimator that breaks 
the correlation and reduces the variance.

Principal Component Analysis
---------------------------------
X_std = StandardScaler().fit_transform(X)
pca = PCA().fit(X_std)
n_components_variance = np.cumsum(pca.explained_variance_ratio_)
# array([0.98198228, 0.99880119, 1.        ])

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)
X_pca_with_constant = sm.add_constant(X_pca)
model = sm.OLS(y, X_pca_with_constant) # sm = statsmodels.api
results = model.fit()
print(results.summary())

plt.scatter(X_pca, y)
plt.plot(X_pca, results.predict(X_pca_with_constant), color = 'blue')
plt.show()
     
Hierarchical clustering 
----------------------------
https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features

Categorical Feature -> pd.get_dummies(), OneHotEncoder()
-------------------

Data Augmentation
-----------------------

# Dealing with Sparse Feature
================================
Sparse features are common in machine learning models, 
especially in the form of one-hot encoding. These features can result 
in issues in machine learning models like overfitting, inaccurate 
feature importances, and high variance. It is recommended that sparse 
features should be pre-processed by methods like feature hashing or 
removing the feature to reduce the negative impacts on the results.


Removing features from the model ->  
--------------------------------------
features with low variance are removed. 
However, sparse features that have important signals should not be 
removed in this process.
LASSO regularization can be used to decrease the number of features. 
Rule-based methods like setting a variance threshold for including 
features in the model might also be useful.

Make the features dense ->
--------------------------
PCA methods can be used to project the features into the directions of 
the principal components and select from the most important components.

feature hashing, sparse features can be binned into the desired number of 
output features using a hash function. Care must be taken to choose a 
generous number of output features to prevent hash collisions.

Some versions of machine learning models are robust towards sparse data 
and may be used instead of changing the dimensionality of the data. 
For example, the entropy-weighted k-means algorithm is 
better suited to this problem than the regular k-means algorithm.


# Dealing with Missing Data
==================================




----------------------------------------------------------------------------
# check residual plot -> check mean & median are how much apart 
==========================================================================
 residuals = y_test -y_test_pred
 sns.distplot(residuals)
 plt.axvline(x=np.mean(residuals),label='mean')
 plt.axvline(x=np.median(residuals),label='median')
 plt.xlabel('Residuals')

=========================================================================
# recursive feature elemination
==========================================================================
from sklearn.feature_selection import RFECV


min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=StratifiedKFold(2),
    scoring="accuracy", # proportion of correct classifications
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores

plt.plot(
    range(min_features_to_select, 
    	len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_
)

==========================================================================
# plot classification probability

========================================
# Bias ~ Variance/Overfit~Underfit
=========================================
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)
common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([naive_bayes, svc]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")



======================================
# normalize time-series
=======================================
from scipy.stats import norm


def get_NormArray(df, n, mode = 'total', linear = False):
'''
         It computes the normalized value on the stats of n values ( Modes: total or scale ) 
         using the formulas from the book "Statistically sound machine learning..."
         (Aronson and Masters) but the decission to apply a non linear scaling is left to the user.
         It is modified to fit the data from -1 to 1 instead of -100 to 100
         df is an imput DataFrame. it returns also a DataFrame, but it could return a list.
         n define the number of data points to get the mean and the quartiles for the normalization
         modes: scale: scale, without centering. total:  center and scale.
 '''
temp =[]

for i in range(len(df))[::-1]:

    if i  >= n: # there will be a traveling norm until we reach the initian n values. 
                # those values will be normalized using the last computed values of F50,F75 and F25
        F50 = df[i-n:i].quantile(0.5)
        F75 =  df[i-n:i].quantile(0.75)
        F25 =  df[i-n:i].quantile(0.25)

    if linear == True and mode == 'total':
         v = 0.5 * ((df.iloc[i]-F50)/(F75-F25))-0.5
    elif linear == True and mode == 'scale':
         v =  0.25 * df.iloc[i]/(F75-F25) -0.5
    elif linear == False and mode == 'scale':
         v = 0.5* norm.cdf(0.25*df.iloc[i]/(F75-F25))-0.5

    else: # even if strange values are given, it will perform full normalization with compression as default
        v = norm.cdf(0.5*(df.iloc[i]-F50)/(F75-F25))-0.5

    temp.append(v[0])
return  pd.DataFrame(temp[::-1])


==========================================================================
# Build Random forest
=============================================
# generate a 1,000 subsets of the training set, each containing a 100 
# instances selected randomly.


rs = ShuffleSplit(n_splits=1000, train_size=100, test_size=0)

decision_trees = list()
ds_test_scores = list()

for train_idxs, _ in rs.split(X_train, y_train):
    
    # get sample
    x_bs = X_train[train_idxs]
    y_bs = y_train[train_idxs]
    
    # train decision tree
    clf = DecisionTreeClassifier(max_leaf_nodes=4)
    clf.fit(x_bs, y_bs)
    decision_trees.append(clf)
    
    # evaluate decision tree
    ds_test_scores.append(clf.score(X_test, y_test))
    
    # delete model
    del(clf)

 # For each test set instance, generate the predictions of the 1,000 
 # decision trees, and keep only the most frequent prediction. 
 # This approach gives you majority-vote predictions over the test set.

from scipy.stats import mode

all_preds = list()
for tree in decision_trees:
    all_preds.append(tree.predict(X_test).tolist())    
trees_preds = np.array(all_preds)
trees_preds.shape



preds, _ = mode(trees_preds, axis=0)

sum(preds.squeeze() == y_test)/len(y_test) # accuracy of random forests

        ThymeBoost model 