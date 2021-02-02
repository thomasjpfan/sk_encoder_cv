from abc import ABC, abstractmethod
import numpy as np

from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from .base import _BaseEncoder
from .base import NestedEncoderCV, BaggingEncoder


class _TargetEncoder(ABC, _BaseEncoder):
    def __init__(self, categories="auto"):
        self.categories = categories

    @abstractmethod
    def _encode_y(self, y):
        """Encode y"""

    def fit(self, X, y):
        """Fit the TargetRegressorEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        self._fit(X, y)
        return self

    def transform(self, X, y=None):
        """Encodes X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        check_is_fitted(self)
        X_int, X_known = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        return self._transform_X_int(X_int, X_known)

    def fit_transform(self, X, y):
        """Fit the encoder and encodes `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X_int, X_known = self._fit(X, y)
        return self._transform_X_int(X_int, X_known)

    def _transform_X_int(self, X_int, X_known):
        """Transform integer encoded X. `X_int` and `X_known` are returned
        by `_transform`.
        """
        out = np.empty_like(X_int, dtype=float)
        for i, cat_encoding in enumerate(self.encodings_):
            np.take(cat_encoding, X_int[:, i], out=out[:, i])
            out[~X_known[:, i], i] = self.encoding_mean_
        return out

    def _fit(self, X, y):
        """Fit the encoder"""
        y = self._encode_y(y)

        fit_results = super()._fit(
            X, handle_unknown="ignore", return_counts=True, force_all_finite="allow-nan"
        )
        X_int, X_known = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        # Makes sure unknown categories are not used fot fitting
        X_int[~X_known] = -1
        self.encoding_mean_ = y_mean = np.mean(y)

        # y is constant the encoding will be the constant
        if np.ptp(y) == 0.0:
            self.encodings_ = [
                np.full(len(cat), fill_value=y_mean, dtype=float)
                for cat in self.categories_
            ]
            return X_int, X_known

        y_variance = np.var(y)

        n_samples, n_features = X_int.shape

        cat_encodings = []
        category_counts = fit_results["category_counts"]

        for i in range(n_features):
            n_cats = len(self.categories_[i])
            cat_means = np.zeros(n_cats, dtype=float)
            cat_var_ratio = np.ones(n_cats, dtype=float)

            for encoding in range(n_cats):
                y_tmp = y[X_int[:, i] == encoding]
                if y_tmp.size:
                    cat_means[encoding] = np.mean(y_tmp)
                    cat_var_ratio[encoding] = np.var(y_tmp)

            # partial-pooling estimates
            cat_counts = category_counts[i]
            cat_var_ratio /= y_variance

            cat_encoded = cat_counts * cat_means + cat_var_ratio * y_mean
            cat_encoded /= cat_counts + cat_var_ratio
            cat_encodings.append(cat_encoded)

        self.encodings_ = cat_encodings
        return X_int, X_known


class TargetRegressorEncoder(_TargetEncoder):
    """Target Encoder for Regression Targets.

    Each category is encoded based on its effect on the target variable. The
    encoding scheme takes a weighted average estimated by a multilevel
    linear model.

    Read more in the :ref:`User Guide <target_regressor_encoder>`.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values in ascending order.

        The used categories can be found in the `categories_` attribute.

    Attributes
    ----------
    encodings_ : list of shape (n_features,) of ndarray
        For feature `i`, `encodings_[i]` is the encoding matching the
        categories listed in `categories_[i]`.

    categories_ : list of shape (n_features,) of ndarray
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of :meth:`transform`).

    encoding_mean_ : float
        The overall mean of the target.

    See Also
    --------
    sklearn.preprocessing.OrdinalEncoder : Performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.preprocessing.OneHotEncoder : Performs a one-hot encoding of
      categorical features.
    """

    def _encode_y(self, y):
        return column_or_1d(y, warn=True)


class TargetClassifierEncoder(_TargetEncoder):
    def _encode_y(self, y):
        return LabelEncoder().fit_transform(y)


class TargetRegressorEncoderCV(NestedEncoderCV):
    def __init__(self, categories="auto", n_jobs=None, cv=5):
        self.categories = categories
        super().__init__(
            n_jobs=n_jobs,
            cv=cv,
            classifier=False,
            encoder=TargetRegressorEncoder(categories=self.categories),
        )


class TargetClassifierEncoderCV(NestedEncoderCV):
    def __init__(self, categories="auto", n_jobs=None, cv=5):
        self.categories = categories
        super().__init__(
            n_jobs=n_jobs,
            cv=cv,
            classifier=True,
            encoder=TargetClassifierEncoder(categories=self.categories),
        )


class TargetClassifierBaggingEncoder(BaggingEncoder):
    def __init__(self, categories="auto", n_jobs=None, cv=5):
        self.categories = categories
        super().__init__(
            n_jobs=n_jobs,
            cv=cv,
            classifier=False,
            encoder=TargetRegressorEncoder(categories=self.categories),
        )


class TargetRegressionBaggingEncoder(BaggingEncoder):
    def __init__(self, categories="auto", n_jobs=None, cv=5):
        self.categories = categories
        super().__init__(
            n_jobs=n_jobs,
            cv=cv,
            classifier=True,
            encoder=TargetClassifierEncoder(categories=self.categories),
        )


class _TargetEncoderBS(ABC, _BaseEncoder):
    """Target Encoder with Bühlmann-Straub estimation."""

    def __init__(self, categories="auto"):
        self.categories = categories

    @abstractmethod
    def _encode_y(self, y):
        """Encode y"""

    def fit(self, X, y):
        """Fit the TargetRegressorEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        self._fit(X, y)
        return self

    def transform(self, X, y=None):
        """Encodes X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        check_is_fitted(self)
        X_int, X_known = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        return self._transform_X_int(X_int, X_known)

    def fit_transform(self, X, y):
        """Fit the encoder and encodes `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X_int, X_known = self._fit(X, y)
        return self._transform_X_int(X_int, X_known)

    def _transform_X_int(self, X_int, X_known):
        """Transform integer encoded X. `X_int` and `X_known` are returned
        by `_transform`.
        """
        out = np.empty_like(X_int, dtype=float)
        for i, cat_encoding in enumerate(self.encodings_):
            np.take(cat_encoding, X_int[:, i], out=out[:, i])
            out[~X_known[:, i], i] = self.encoding_mean_
        return out

    def _fit(self, X, y):
        """Fit the encoder"""
        y = self._encode_y(y)

        fit_results = super()._fit(
            X, handle_unknown="ignore", return_counts=True, force_all_finite="allow-nan"
        )
        X_int, X_known = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        # Makes sure unknown categories are not used fot fitting
        X_int[~X_known] = -1
        self.encoding_mean_ = y_mean = np.mean(y)

        # if y is constant the encoding will be the constant
        if np.ptp(y) == 0.0:
            self.encodings_ = [
                np.full(len(cat), fill_value=y_mean, dtype=float)
                for cat in self.categories_
            ]
            return X_int, X_known

        y_mean = np.mean(y)

        n_samples, n_features = X_int.shape

        cat_encodings = []

        for i in range(n_features):
            n_cats = len(self.categories_[i])
            if n_cats <= 2:
                cat_encodings.append(np.arange(n_cats).astype(float))
                continue
            cat_means = np.zeros(n_cats, dtype=float)
            cat_var = np.ones(n_cats, dtype=float)
            # number of samples having this level=encoding
            cat_weight = fit_results["category_counts"][i]

            for encoding in range(n_cats):
                y_tmp = y[X_int[:, i] == encoding]
                if y_tmp.size:
                    cat_means[encoding] = np.mean(y_tmp)
                    if y_tmp.size <= 2:
                        cat_var[encoding] = np.nan
                    else:
                        cat_var[encoding] = np.var(y_tmp, ddof=1)

            # Use Bühlmann-Straub estimators
            # sigma^2: same for all levels
            sigma2 = np.nanmean(cat_var)

            # tau^2
            tau2 = np.average((cat_means - y_mean) ** 2, weights=cat_weight)
            tau2 *= n_cats / (n_cats - 1)
            # tau2_biased = tau2
            tau2 -= n_cats * sigma2 / np.sum(cat_weight)
            c = 1.0 / np.average(
                1 - cat_weight / np.sum(cat_weight), weights=cat_weight
            )
            c *= (n_cats - 1) / n_cats
            tau2 *= c
            tau2 = max(0, tau2)
            # Instead of tau2 = 0, we use a biased estimator instead
            # if tau2 == 0:
            #     tau2 = c * tau2_biased

            if tau2 == 0 or np.isnan(sigma2):
                alpha = np.zeros_like(cat_means)
            else:
                alpha = cat_weight / (cat_weight + sigma2 / tau2)

            # homogeneous Bühlmann-Straub estimator,
            # homogeneous means, we use the following mean estimate of y.
            if tau2 == 0 or np.isnan(sigma2):
                y_mean_hom = y_mean
            else:
                y_mean_hom = np.average(cat_means, weights=alpha)

            cat_encoded = alpha * cat_means + (1.0 - alpha) * y_mean_hom
            cat_encodings.append(cat_encoded)

        self.encodings_ = cat_encodings
        return X_int, X_known


class TargetRegressorEncoderBS(_TargetEncoderBS):
    """Target Encoder for Regression Targets a la Bühlmann Straub.

    See Also
    --------
    TargetRegressorEncoder
    """

    def _encode_y(self, y):
        return column_or_1d(y, warn=True)


class TargetClassifierEncoderBS(_TargetEncoderBS):
    def _encode_y(self, y):
        return LabelEncoder().fit_transform(y)


class TargetClassifierBaggingEncoderBS(BaggingEncoder):
    def __init__(self, categories="auto", n_jobs=None, cv=5):
        self.categories = categories
        super().__init__(
            n_jobs=n_jobs,
            cv=cv,
            classifier=False,
            encoder=TargetRegressorEncoderBS(categories=self.categories),
        )


class TargetRegressionBaggingEncoderBS(BaggingEncoder):
    def __init__(self, categories="auto", n_jobs=None, cv=5):
        self.categories = categories
        super().__init__(
            n_jobs=n_jobs,
            cv=cv,
            classifier=True,
            encoder=TargetClassifierEncoderBS(categories=self.categories),
        )
