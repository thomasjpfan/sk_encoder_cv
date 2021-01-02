from pprint import pprint
import json
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from dataclasses import dataclass
from typing import List
from typing import Optional


@dataclass
class DataInfo:
    data_name: str
    data_id: int
    is_classification: bool
    columns_to_remove: Optional[List[str]] = None


def fetch_openml_and_clean(data_info: DataInfo):
    print(f"fetching and loading {data_info.data_name} dataset from openml")
    X, y = fetch_openml(data_id=data_info.data_id, return_X_y=True, as_frame=True)

    if data_info.columns_to_remove:
        X = X.drop(data_info.columns_to_remove, axis="columns")

    if data_info.is_classification:
        y = LabelEncoder().fit_transform(y)

    return X, y


def get_estimator(encoder, data_info):
    if data_info.is_classification:
        HistGradientEst = HistGradientBoostingClassifier
    else:
        HistGradientEst = HistGradientBoostingRegressor

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

    return Pipeline([("prep", prep), ("est", HistGradientEst(random_state=42))])


def write_results(results, results_dir, data_info, encoder_str, cv, write_results):
    # process results
    output = {}
    for key, value in results.items():
        output[f"{key}_mean"] = value.mean()
        output[f"{key}_std"] = value.std()
    output["encoder"] = encoder_str
    output["data_id"] = data_info.data_id
    output["data_name"] = data_info.data_name
    output["is_classification"] = data_info.is_classification
    output["cv"] = cv

    pprint(output)

    if write_results:
        results_path = results_dir / f"{data_info.data_name}_{encoder_str}.json"

        with results_path.open("w") as f:
            json.dump(output, f)
        print(f"Wrote results to {results_path}")
