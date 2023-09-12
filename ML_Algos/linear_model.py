The sklearn.linear_model module implements a variety of linear models. 
===============================================================================
# 1. Linear classifiers
# 2. Classical linear regressors
# 3. Regressors with variable selection - The following estimators have built-in 
# variable selection fitting procedures, but any estimator using a L1 or elastic-net 
# penalty also performs variable selection: typically SGDRegressor or SGDClassifier 
# with an appropriate penalty.
# 4. Bayesian regressors
# 5. Multi-task linear regressors with variable selection - These estimators fit 
# multiple regression problems (or tasks) jointly, while inducing sparse coefficients. 
# While the inferred coefficients may differ between the tasks, they are constrained 
# to agree on the features that are selected (non-zero coefficients).
# 6. Outlier-robust regressors - Any estimator using the Huber loss would also be robust 
# to outliers, e.g. SGDRegressor with loss='huber'.
# 7. Generalized linear models (GLM) for regression - These models allow for response 
# variables to have error distributions other than a normal distribution
# 8. Miscellaneous
======================================================================================
# Basic Implementation
# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
# # calculate coefficients
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# b0, b1 = coefficients(dataset)
# print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1)

def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
	train, test = train_test_split(dataset, split) #check helper.py 
	# test contains dataset_copy
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args) # here simple_linear_regression
	actual = [row[-1] for row in test]
	rmse = rmse_metric(actual, predicted)
	return rmse


# Simple linear regression on insurance dataset
seed(1)
# load and prepare data
filename = 'insurance.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('RMSE: %.3f' % (rmse)

======================================================================================

class sklearn.linear_model.LinearRegression(*, fit_intercept=True, 
	normalize='deprecated', copy_X=True, n_jobs=None, positive=False)


    # Ordinary least squares Linear Regression.
    # LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    # to minimize the residual sum of squares between the observed targets in
    # the dataset, and the targets predicted by the linear approximation.

    # The least squares solution is computed using the singular value decomposition 
    # of X. If X is a matrix of shape (n_samples, n_features) this method has a cost 
    # of O(n_samples*n_features*n_features), assuming that n_samples >= n_features.

    # Parameters
    # ----------
    # fit_intercept : bool, default=True
    #     Whether to calculate the intercept for this model. If set
    #     to False, no intercept will be used in calculations
    #     (i.e. data is expected to be centered).
    # normalize : bool, default=False
    #     This parameter is ignored when ``fit_intercept`` is set to False.
    #     If True, the regressors X will be normalized before regression by
    #     subtracting the mean and dividing by the l2-norm.
    #     If you wish to standardize, please use
    #     :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
    #     on an estimator with ``normalize=False``.
    #     .. deprecated:: 1.0
    #        `normalize` was deprecated in version 1.0 and will be
    #        removed in 1.2.
    # copy_X : bool, default=True
    #     If True, X will be copied; else, it may be overwritten.
    # n_jobs : int, default=None
    #     The number of jobs to use for the computation. This will only provide
    #     speedup in case of sufficiently large problems, that is if firstly
    #     `n_targets > 1` and secondly `X` is sparse or if `positive` is set
    #     to `True`. ``None`` means 1 unless in a
    #     :obj:`joblib.parallel_backend` context. ``-1`` means using all
    #     processors. See :term:`Glossary <n_jobs>` for more details.
    # positive : bool, default=False
    #     When set to ``True``, forces the coefficients to be positive. This
    #     option is only supported for dense arrays.
    #     .. versionadded:: 0.24
    # Attributes
    # ----------
    # coef_ : array of shape (n_features, ) or (n_targets, n_features)
    #     Estimated coefficients for the linear regression problem.
    #     If multiple targets are passed during the fit (y 2D), this
    #     is a 2D array of shape (n_targets, n_features), while if only
    #     one target is passed, this is a 1D array of length n_features.
    # rank_ : int
    #     Rank of matrix `X`. Only available when `X` is dense.
    # singular_ : array of shape (min(X, y),)
    #     Singular values of `X`. Only available when `X` is dense.
    # intercept_ : float or array of shape (n_targets,)
    #     Independent term in the linear model. Set to 0.0 if
    #     `fit_intercept = False`.
    # n_features_in_ : int
    #     Number of features seen during :term:`fit`.
    #     .. versionadded:: 0.24
    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     Names of features seen during :term:`fit`. Defined only when `X`
    #     has feature names that are all strings.
    #     .. versionadded:: 1.0
    # See Also
    # --------
    # Ridge : Ridge regression addresses some of the
    #     problems of Ordinary Least Squares by imposing a penalty on the
    #     size of the coefficients with l2 regularization.
    # Lasso : The Lasso is a linear model that estimates
    #     sparse coefficients with l1 regularization.
    # ElasticNet : Elastic-Net is a linear regression
    #     model trained with both l1 and l2 -norm regularization of the
    #     coefficients.
    # Notes
    # -----
    # From the implementation point of view, this is just plain Ordinary
    # Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
    # (scipy.optimize.nnls) wrapped as a predictor object.
    Examples
    --------
    import numpy as np
	import pandas as pd
	from sklearn.linear_model import LinearRegression
	X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
	# y = 1 * x_0 + 2 * x_1 + 3
	y = np.dot(X, np.array([1, 2])) + 3
	X = pd.DataFrame(X, columns=[f"Features {i}" for i in range(X.shape[1])])
	print("X ",X,end='\n')
	reg = LinearRegression(fit_intercept=False,n_jobs=2,
				positive=False,copy_X=False).fit(X, y)
	print("Score ",reg.score(X, y),end="\n") 
	print("Coeff ",reg.coef_,end="\n")
	print("Intercept ",reg.intercept_,end="\n")
	print("Predict ",reg.predict(np.array([[3, 5]])),end="\n")
	print("n_features_in_ ",reg.n_features_in_,end="\n")
	print("rank_ ",reg.rank_,end="\n")
	print("singular_ ",reg.singular_,end="\n")
	print("feature_names_in_ ",reg.feature_names_in_,end="\n")
	print("X ",X,end='\n')
	# 	X     Features 0  Features 1
	# 0           1           1
	# 1           1           2
	# 2           2           2
	# 3           2           3
	# Score  0.7482517482517481
	# Coeff  [2.09090909 2.54545455]
	# Intercept  0.0
	# Predict  [19.]
	# n_features_in_  2
	# rank_  2
	# singular_  [5.25371017 0.63129192]
	# feature_names_in_  ['Features 0' 'Features 1']
	# X     Features 0  Features 1
	# 0           1           1
	# 1           1           2
	# 2           2           2
	# 3           2           3

	reg = LinearRegression(fit_intercept=True,copy_X=True,positive=False).fit(X, y)
	# 	X     Features 0  Features 1
	# 0           1           1
	# 1           1           2
	# 2           2           2
	# 3           2           3
	# Score  1.0
	# Coeff  [1. 2.]
	# Intercept  3.0000000000000018
	# Predict  [16.]
	# n_features_in_  2
	# rank_  2
	# singular_  [1.61803399 0.61803399]
	# feature_names_in_  ['Features 0' 'Features 1']
	# X     Features 0  Features 1
	# 0           1           1
	# 1           1           2
	# 2           2           2
	# 3           2           3



Methods
------------
get_params([deep]) -Get parameters for this estimator.
set_params(**params) - Set the parameters of this estimator.
predict(X) - Predict using the linear model.
score(X, y[, sample_weight]) -Return the coefficient of determination of the 
								prediction.
fit(X, y[, sample_weight]) - Fit linear model.
====================================================================================
# WorkFlow

# TODO: bayesian_ridge_regression and bayesian_regression_ard
# should be squashed into its respective objects.

SPARSE_INTERCEPT_DECAY = 0.01
# For sparse data intercept updates are scaled by this decay factor to avoid
# intercept oscillation.

# FIXME in 1.2: parameter 'normalize' should be removed from linear models
# in cases where now normalize=False. The default value of 'normalize' should
# be changed to False in linear models where now normalize=True
def _deprecate_normalize(normalize, default, estimator_name):
	"""Normalize is to be deprecated from linear models and a use of
    a pipeline with a StandardScaler is to be recommended instead.
    Here the appropriate message is selected to be displayed to the user
    depending on the default normalize value (as it varies between the linear
    models and normalize value selected by the user).
    Parameters
    ----------
    normalize : bool,
        normalize value passed by the user
    default : bool,
        default normalize value used by the estimator
    estimator_name : str
        name of the linear estimator which calls this function.
        The name will be used for writing the deprecation warnings
    Returns
    -------
    normalize : bool,
        normalize value which should further be used by the estimator at this
        stage of the depreciation process
    Notes
    -----
    This function should be updated in 1.2 depending on the value of
    `normalize`:
    - True, warning: `normalize` was deprecated in 1.2 and will be removed in
      1.4. Suggest to use pipeline instead.
    - False, `normalize` was deprecated in 1.2 and it will be removed in 1.4.
      Leave normalize to its default value.
    - `deprecated` - this should only be possible with default == False as from
      1.2 `normalize` in all the linear models should be either removed or the
      default should be set to False.
    This function should be completely removed in 1.4.
    """
def make_dataset(X, y, sample_weight, random_state=None):
	"""Create ``Dataset`` abstraction for sparse and dense inputs.
    This also returns the ``intercept_decay`` which is different
    for sparse datasets.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples, )
        Target values.
    sample_weight : numpy array of shape (n_samples,)
        The weight of each sample
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset random sampling. It is not
        used for dataset shuffling.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    dataset
        The ``Dataset`` abstraction
    intercept_decay
        The intercept decay
    """


# TODO: _rescale_data should be factored into _preprocess_data.
# Currently, the fact that sag implements its own way to deal with
# sample_weight makes the refactoring tricky.


def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.
    For many linear models, this enables easy support for sample_weight because
        (y - X w)' S (y - X w)
    with S = diag(sample_weight) becomes
        ||y_rescaled - X_rescaled w||_2^2
    when setting
        y_rescaled = sqrt(S) y
        X_rescaled = sqrt(S) X
    Returns
    -------
    X_rescaled : {array-like, sparse matrix}
    y_rescaled : {array-like, sparse matrix}
    """

# ABCMeta metaclass provides a method called register method that can be 
# invoked by its instance. By using this register method, any abstract base class 
# can become an ancestor of any arbitrary concrete class.

class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""
    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def predict(self, X):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
     def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""
     def _more_tags(self):
        return {"requires_y": True}

# XXX Should this derive from LinearModel? It should be a mixin, not an ABC.
# Maybe the n_features checking can be moved to LinearModel.
class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.
    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.
        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """

class SparseCoefMixin:
    """Mixin for converting coef_ to and from CSR format.
    L1-regularizing estimators should inherit this.
    """

    def densify(self):
        """
        Convert coefficient matrix to dense array format.
        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.
        Returns
        -------
        self
            Fitted estimator.
        """
    def sparsify(self):
        """
        Convert coefficient matrix to sparse format.
        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.
        The ``intercept_`` member is not converted.
        Returns
        -------
        self
            Fitted estimator.
        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.
        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.
        """
        msg = "Estimator, %(name)s, must be fitted before sparsifying."
        check_is_fitted(self, msg=msg)
        self.coef_ = sp.csr_matrix(self.coef_)
        return 

class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.
    """
    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : object
            Fitted Estimator.
        """
def _check_precomputed_gram_matrix(
X, precompute, X_offset, X_scale, rtol=None, atol=1e-5
):
"""Computes a single element of the gram matrix and compares it to
the corresponding element of the user supplied gram matrix.
If the values do not match a ValueError will be thrown.
Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Data array.
precompute : array-like of shape (n_features, n_features)
    User-supplied gram matrix.
X_offset : ndarray of shape (n_features,)
    Array of feature means used to center design matrix.
X_scale : ndarray of shape (n_features,)
    Array of feature scale factors used to normalize design matrix.
rtol : float, default=None
    Relative tolerance; see numpy.allclose
    If None, it is set to 1e-4 for arrays of dtype numpy.float32 and 1e-7
    otherwise.
atol : float, default=1e-5
    absolute tolerance; see :func`numpy.allclose`. Note that the default
    here is more tolerant than the default for
    :func:`numpy.testing.assert_allclose`, where `atol=0`.
Raises
------
ValueError
    Raised when the provided Gram matrix is not consistent.
"""
def _pre_fit(
    X,
    y,
    Xy,
    precompute,
    normalize,
    fit_intercept,
    copy,
    check_input=True,
    sample_weight=None,
):
    """Function used at beginning of fit in linear models with L1 or L0 penalty.
    This function applies _preprocess_data and additionally computes the gram matrix
    `precompute` as needed as well as `Xy`.
    Parameters
    ----------
    order : 'F', 'C' or None, default=None
        Whether X and y will be forced to be fortran or c-style. Only relevant
        if sample_weight is not None.
    """

=====================================================================================
class BaseEstimator:
    """Base class for all estimators in scikit-learn.
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


=================================================================================

class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""
    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        # from ..utils.extmath import safe_sparse_dot
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def predict(self, X):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.0




# XXX Should this derive from LinearModel? It should be a mixin, not an ABC.
# Maybe the n_features checking can be moved to LinearModel.
class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.
    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.
        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        # Return : Flattened array having same type as the Input array and and order as per choice.
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
       #  	    Syntax : numpy.vstack(tup)
			    # Parameters :
			    # tup : [sequence of ndarrays] Tuple containing arrays to be stacked. 
			    # The arrays must have the same shape along all but the first axis.
			    # Return : [stacked ndarray] The stacked array of the input arrays.
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob



============================================================================
class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction.
        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        from .metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}


=====================================================================================
class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):

    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        ##### def _deprecate_normalize(normalize, default, estimator_name):
	    """Normalize is to be deprecated from linear models and a use of
	    a pipeline with a StandardScaler is to be recommended instead.
	    Here the appropriate message is selected to be displayed to the user
	    depending on the default normalize value (as it varies between the linear
	    models and normalize value selected by the user).
	    Parameters
	    ----------
	    normalize : bool,
	        normalize value passed by the user
	    default : bool,
	        default normalize value used by the estimator
	    estimator_name : str
	        name of the linear estimator which calls this function.
	        The name will be used for writing the deprecation warnings
	    Returns
	    -------
	    normalize : bool,
	        normalize value which should further be used by the estimator at this
	        stage of the depreciation process

	    """
        _normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=self.__class__.__name__
        )

        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ["csr", "csc", "coo"]
        # defined inside class BaseEstimator -> Base class for all estimators in scikit-learn.
        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )

        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype, only_non_negative=True
        )

        # X_offset : is calculated using _incremental_mean_and_var if normalized 
        # otherwise np.average(X, axis=0, weights=sample_weight)
        # X_offset = X_offset.astype(X.dtype, copy=False)
        # X -= X_offset
        # X_scale = np.sqrt(X_var, out=X_var)
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=_normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        # Sample weight can be implemented via a simple rescaling.
        X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)

        if self.positive:
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                # This is a wrapper for a FORTRAN non-negative least squares solver.
                # from joblib import Parallel
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            #Common interface for performing matrix vector products
            #scipy.sparse.linalg.LinearOperator
            def matvec(b):
                return X.dot(b) - sample_weight_sqrt * b.dot(X_offset_scale)

            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * b.dot(sample_weight_sqrt)

            X_centered = sparse.linalg.LinearOperator(
                shape=X.shape, matvec=matvec, rmatvec=rmatvec
            )

            if y.ndim < 2:
                self.coef_ = lsqr(X_centered, y)[0]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                # Find the least-squares solution to a large, sparse, linear system of equations.
                # scipy.sparse.linalg.lsqr
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
      #   	From the implementation point of view, this is just plain Ordinary
		    # Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
		    # (scipy.optimize.nnls) wrapped as a predictor object.
            self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


