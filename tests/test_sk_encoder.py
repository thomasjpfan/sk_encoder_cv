from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline

from sk_encoder_cv import (
    TargetClassifierEncoder,
    TargetClassifierEncoderCV,
    TargetRegressorEncoder,
    TargetRegressorEncoderCV,
    TargetRegressorEncoderBS,
    TargetRegressionBaggingEncoder,
    TargetRegressionBaggingEncoderBS,
)
from category_encoders import JamesSteinEncoder
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(".") / "data"


@pytest.mark.parametrize(
    "encoder",
    [TargetClassifierEncoder(), TargetClassifierEncoderCV(), JamesSteinEncoder()],
)
def test_adult(encoder):
    """Smoke test for adult dataset."""

    adult_path = DATA_PATH / "adult.csv"
    adult_df = pd.read_csv(adult_path)

    X = adult_df.drop("class", axis=1)
    y = LabelEncoder().fit_transform(adult_df["class"])

    prep = ColumnTransformer(
        [
            (
                "cat",
                encoder,
                make_column_selector(dtype_include=["object", "category"]),
            ),
            ("num", "passthrough", make_column_selector(dtype_include="number")),
        ]
    )

    hist = Pipeline(
        [("prep", prep), ("est", HistGradientBoostingClassifier(random_state=42))]
    )

    hist.fit(X, y)


@pytest.mark.parametrize(
    "encoder",
    [
        TargetRegressorEncoder(),
        TargetRegressorEncoderCV(),
        TargetRegressorEncoderBS(),
        JamesSteinEncoder(),
        TargetRegressionBaggingEncoder(),
        TargetRegressionBaggingEncoderBS(),
    ],
)
def test_ames(encoder):
    """Smoke test for adult dataset."""

    ames_path = DATA_PATH / "ames.csv"
    ames_df = pd.read_csv(ames_path)

    X = ames_df.drop("SalePrice", axis=1)
    y = ames_df["SalePrice"]

    prep = ColumnTransformer(
        [
            (
                "cat",
                encoder,
                make_column_selector(dtype_include=["object", "category"]),
            ),
            ("num", "passthrough", make_column_selector(dtype_include="number")),
        ]
    )

    hist = Pipeline(
        [("prep", prep), ("est", HistGradientBoostingRegressor(random_state=42))]
    )

    hist.fit(X, y)
