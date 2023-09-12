
# # Calculate the mean value of a list of numbers
# def mean(values):
# 	return sum(values) / float(len(values))
# # Calculate covariance between x and y
# def covariance(x, mean_x, y, mean_y):
# 	covar = 0.0
# 	for i in range(len(x)):
# 		covar += (x[i] - mean_x) * (y[i] - mean_y)
# 	return covar
# # Calculate the variance of a list of numbers
# def variance(values, mean):
# 	return sum([(x-mean)**2 for x in values])

# universal functions - not defined under any class but for sklearn.model

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
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y, sample_weight_sqrt


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    sample_weight=None,
    check_input=True,
):
    """Center and scale data.
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    fit_intercept=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
        If normalize is True, then X_out is rescaled (dense and sparse case)
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Likely performed inplace on input y.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    X_scale : ndarray of shape (n_features,)
        The standard deviation per column of input X.
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order="K")

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            if normalize:
                X_offset, X_var, _ = _incremental_mean_and_var(
                    X,
                    last_mean=0.0,
                    last_variance=0.0,
                    last_sample_count=0.0,
                    sample_weight=sample_weight,
                )
            else:
                X_offset = np.average(X, axis=0, weights=sample_weight)

            X_offset = X_offset.astype(X.dtype, copy=False)
            X -= X_offset

        if normalize:
            X_var = X_var.astype(X.dtype, copy=False)
            # Detect constant features on the computed variance, before taking
            # the np.sqrt. Otherwise constant features cannot be detected with
            # sample weights.
            constant_mask = _is_constant_feature(X_var, X_offset, X.shape[0])
            if sample_weight is None:
                X_var *= X.shape[0]
            else:
                X_var *= sample_weight.sum()
            X_scale = np.sqrt(X_var, out=X_var)
            X_scale[constant_mask] = 1.0
            if sp.issparse(X):
                inplace_column_scale(X, 1.0 / X_scale)
            else:
                X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


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
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y, sample_weight_sqrt

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

    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)

    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]

    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]

    dtypes = [precompute.dtype, expected.dtype]
    if rtol is None:
        rtols = [1e-4 if dtype == np.float32 else 1e-7 for dtype in dtypes]
        rtol = max(rtols)

    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(
            "Gram matrix passed in via 'precompute' parameter "
            "did not pass validation when a single element was "
            "checked - please check that it was computed "
            f"properly. For element ({f1},{f2}) we computed "
            f"{expected} but the user-supplied value was "
            f"{actual}."
        )


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
    n_samples, n_features = X.shape

    if sparse.isspmatrix(X):
        # copy is not needed here as X is not modified inplace when X is sparse
        precompute = False
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy=False,
            check_input=check_input,
            sample_weight=sample_weight,
        )
    else:
        # copy was done in fit if necessary
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy=copy,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # Rescale only in dense case. Sparse cd solver directly deals with
        # sample_weight.
        if sample_weight is not None:
            # This triggers copies anyway.
            X, y, _ = _rescale_data(X, y, sample_weight=sample_weight)

    # FIXME: 'normalize' to be removed in 1.2
    if hasattr(precompute, "__array__"):
        if (
            fit_intercept
            and not np.allclose(X_offset, np.zeros(n_features))
            or normalize
            and not np.allclose(X_scale, np.ones(n_features))
        ):
            warnings.warn(
                "Gram matrix was provided but X was centered to fit "
                "intercept, or X was normalized : recomputing Gram matrix.",
                UserWarning,
            )
            # recompute Gram
            precompute = "auto"
            Xy = None
        elif check_input:
            # If we're going to use the user's precomputed gram matrix, we
            # do a quick check to make sure its not totally bogus.
            _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)

    # precompute if n_samples > n_features
    if isinstance(precompute, str) and precompute == "auto":
        precompute = n_samples > n_features

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype, order="C")
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, "__array__"):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, "__array__") and Xy is None:
        common_dtype = np.find_common_type([X.dtype, y.dtype], [])
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order="C")
            np.dot(X.T, y, out=Xy)
        else:
            # Make sure that Xy is always F contiguous even if X or y are not
            # contiguous: the goal is to make it fast to extract the data for a
            # specific target.
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype, order="F")
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy
