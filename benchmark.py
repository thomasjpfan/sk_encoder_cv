from pprint import pprint
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

from pathlib import Path
from bench_utils import DataInfo
from bench_utils import get_estimator
from bench_utils import format_results
from bench_utils import get_results_path
from bench_utils import load_data
from sklearn.model_selection import cross_validate

from category_encoders import JamesSteinEncoder
from sk_encoder_cv import NestedEncoderCV
from sk_encoder_cv import TargetRegressorEncoder, TargetRegressorEncoderBS
from sk_encoder_cv import TargetRegressorEncoderCV


DATASET_NAMES = [
    "telco",
    "amazon_access",
    "kicks",
    # "taxi",
    # "ames",
    "churn",
    # "adult",
    "dresses_sales",
    "phishing_websites",
]

RESULTS_DIR = Path(".") / "results"
DATA_DIR = Path(".") / "data"
DATA_INFOS = {
    "kicks": DataInfo(
        data_name="kicks",
        data_id=41162,
        is_classification=True,
    ),
    "amazon_access": DataInfo(
        data_name="amazon_access",
        data_id=4135,
        is_classification=True,
    ),
    "telco": DataInfo(
        data_name="telco",
        data_id=42178,
        is_classification=True,
    ),
    "adult": DataInfo(
        data_name="adult",
        data_id=179,
        is_classification=True,
    ),
    "ames": DataInfo(
        data_name="ames",
        data_id=42165,
        is_classification=False,
        columns_to_remove=["Id"],
    ),
    "taxi": DataInfo(
        data_name="taxi",
        data_id=42729,
        is_classification=False,
    ),
    "churn": DataInfo(
        data_name="churn",
        data_id=40701,
        is_classification=True,
    ),
    "dresses_sales": DataInfo(
        data_name="dresses_sales",
        data_id=23381,
        is_classification=True,
    ),
    "phishing_websites": DataInfo(
        data_name="phishing_websites",
        data_id=4534,
        is_classification=True,
    ),
}

ENCODERS = {
    "drop": "drop",
    "SKOrdinalEncoder": make_pipeline(
        SimpleImputer(strategy="constant", fill_value="sk_missing"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    ),
    "SKTargetEncoder": TargetRegressorEncoder(),
    "SKTargetEncoderCV": TargetRegressorEncoderCV(),
    "SKTargetEncoderBS": TargetRegressorEncoderBS(),
    "JamesSteinEncoder": JamesSteinEncoder(),
    "JamesSteinEncoderCV": NestedEncoderCV(JamesSteinEncoder()),
}


def run_single_benchmark(data_str, cv, n_jobs, write_result, force):
    print(f"running benchmark for {data_str}")
    data_info = DATA_INFOS[data_str]

    results_path = get_results_path(RESULTS_DIR, data_info)
    if results_path.exists() and not force:
        print(
            f"benchmark for {data_str} already exist in {results_path} exists "
            "pass --force to rerun"
        )
        return

    meta_data = load_data(data_info=data_info)
    X, y = meta_data["X"], meta_data["y"]
    if data_info.is_classification:
        scoring = ["accuracy", "roc_auc", "average_precision"]
    else:
        scoring = ["r2", "neg_mean_squared_error", "neg_median_absolute_error"]

    all_results = []

    for encoder_str, encoder in ENCODERS.items():
        print(f"running benchmark for {data_str} for {encoder_str}")
        if (
            meta_data["categorical features"] == meta_data["n_features"]
            and encoder_str == "drop"
        ):
            print("Skipping drop because all features are categorical")
            continue
        estimator = get_estimator(encoder, data_info)

        results = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            scoring=scoring,
        )
        formated_results = format_results(
            results, data_info, encoder_str, cv, meta_data
        )
        pprint(formated_results)
        all_results.append(formated_results)

    if write_result:
        results_df = pd.DataFrame.from_records(all_results)
        print(f"Wrote results to {results_path}")
        results_df.to_csv(results_path, index=False)


def _run_single_benchmark(args):
    run_single_benchmark(
        data_str=args.dataset,
        cv=args.cv,
        n_jobs=args.n_jobs,
        write_result=not args.no_write,
        force=args.force,
    )


def _run_all_benchmark(args):
    print("running all benchmarks")
    for data_str in DATA_INFOS:
        if data_str not in DATASET_NAMES:
            continue
        run_single_benchmark(
            data_str, args.cv, args.n_jobs, not args.no_write, args.force
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--cv", default=5, type=int)
    parser.add_argument("--n-jobs", default=1, type=int)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--force", action="store_true")

    subparsers = parser.add_subparsers()

    single_parser = subparsers.add_parser("single", help="Run single benchmark")
    single_parser.add_argument("dataset", choices=DATA_INFOS)

    single_parser.set_defaults(func=_run_single_benchmark)

    run_all_parser = subparsers.add_parser("all", help="Run all benchmarks")
    run_all_parser.set_defaults(func=_run_all_benchmark)

    args = parser.parse_args()
    args.func(args)
