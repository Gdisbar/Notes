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