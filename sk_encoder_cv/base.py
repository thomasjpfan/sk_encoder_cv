from abc import ABC
import numpy as np

from sklearn.utils.validation import check_array
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.base import clone

from .utils import _encode, _check_unknown, _unique, _get_counts
from .utils import cross_val_transform


class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _check_X(self, X, force_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, eg for the `categories_` attribute.

        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = force_all_finite

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = self._get_feature(X, feature_idx=i)
            Xi = check_array(
                Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _get_feature(self, X, feature_idx):
        if hasattr(X, "iloc"):
            # pandas dataframes
            return X.iloc[:, feature_idx]
        # numpy arrays, sparse arrays
        return X[:, feature_idx]

    def _fit(
        self, X, handle_unknown="error", force_all_finite=True, return_counts=False
    ):
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )

        if self.categories != "auto":
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if categories is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []
        category_counts = []

        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == "auto":
                result = _unique(Xi, return_counts=return_counts)
                if return_counts:
                    cats, counts = result
                    category_counts.append(counts)
                else:
                    cats = result
            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype.kind not in "OU":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not "
                        "supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]) or (
                        np.isnan(sorted_cats[-1]) and not np.isnan(sorted_cats[-1])
                    ):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
                if return_counts:
                    category_counts.append(_get_counts(Xi, cats))

            self.categories_.append(cats)

        return {"category_counts": category_counts, "n_samples": n_samples}

    def _transform(self, X, handle_unknown="error", force_all_finite=True):
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )

        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)

        if n_features != len(self.categories_):
            raise ValueError(
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {} features "
                "and the X has {} features.".format(
                    len(
                        self.categories_,
                    ),
                    n_features,
                )
            )

        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)

            if not np.all(valid_mask):
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (
                        self.categories_[i].dtype.kind in ("U", "S")
                        and self.categories_[i].itemsize > Xi.itemsize
                    ):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    else:
                        Xi = Xi.copy()

                    Xi[~valid_mask] = self.categories_[i][0]
            # We use check_unknown=False, since _check_unknown was
            # already called above.
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)

        return X_int, X_mask

    def _more_tags(self):
        return {"X_types": ["categorical"]}


class NestedEncoderCV(ABC, TransformerMixin, BaseEstimator):
    def __init__(self, encoder, n_jobs=None, cv=5, classifier=False):
        self.encoder = encoder
        self.n_jobs = n_jobs
        self.cv = cv
        self.classifier = classifier

    def fit_transform(self, X, y):
        output = cross_val_transform(
            self.encoder,
            X,
            y,
            cv=self.cv,
            classifier=self.classifier,
            n_jobs=self.n_jobs,
        )
        self.fit(X, y)
        return output

    def transform(self, X, y=None):
        return self.encoder_.transform(X, y=None)

    def fit(self, X, y):
        # should never be called; warn here?
        self.encoder_ = clone(self.encoder)
        self.encoder_.fit(X, y)
        return self
