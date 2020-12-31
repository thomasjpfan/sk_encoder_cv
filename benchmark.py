from itertools import product
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from pathlib import Path
from bench_utils import DataInfo
from bench_utils import fetch_openml_and_clean
from bench_utils import get_estimator
from bench_utils import make_target_encoder
from bench_utils import write_results
from sklearn.model_selection import cross_validate


RESULTS_DIR = Path(".") / "results"
DATA_INFOS = {
    "adult": DataInfo(
        data_name="adult", data_id=179, is_classifier=True, target_column="class"
    ),
    "ames": DataInfo(
        data_name="ames",
        data_id=42165,
        is_classifier=False,
        target_column="SalePrice",
        columns_to_remove=["Id"],
    ),
}

ENCODERS = {
    "SKTargetEncoder": "SKTargetEncoder",
    "SKTargetEncoderCV": "SKTargetEncoderCV",
    "OrdinalEncoder": make_pipeline(
        SimpleImputer(strategy="constant", fill_value="sk_missing"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    ),
    "drop": "drop",
}


def run_single_benchmark(data_str, encoder_str, cv, n_jobs):
    print(f"running benchmark for {data_str} and {encoder_str}")
    data_info = DATA_INFOS[data_str]
    encoder = make_target_encoder(ENCODERS[encoder_str], data_info)

    X, y = fetch_openml_and_clean(data_info=data_info)
    estimator = get_estimator(encoder, data_info)

    if data_info.is_classifier:
        scoring = ["average_precision", "roc_auc", "accuracy"]
    else:
        scoring = ["neg_mean_squared_error", "r2"]

    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        scoring=scoring,
    )
    write_results(results, RESULTS_DIR, data_info, encoder_str, cv)


def _run_single_benchmark(args):
    run_single_benchmark(
        data_str=args.dataset,
        encoder_str=args.encoder,
        cv=args.cv,
        n_jobs=args.n_jobs,
    )


def _run_all_benchmark(args):
    print("running all benchmarks")
    for data_str, encoder_str in product(DATA_INFOS, ENCODERS):
        run_single_benchmark(data_str, encoder_str, args.cv, args.n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--cv", default=5, type=int)
    parser.add_argument("--n-jobs", default=1, type=int)

    subparsers = parser.add_subparsers()

    single_parser = subparsers.add_parser("single", help="Run single benchmark")
    single_parser.add_argument("dataset", choices=DATA_INFOS)
    single_parser.add_argument("encoder", choices=ENCODERS)

    single_parser.set_defaults(func=_run_single_benchmark)

    run_all_parser = subparsers.add_parser("all", help="Run all benchmarks")
    run_all_parser.set_defaults(func=_run_all_benchmark)

    args = parser.parse_args()
    args.func(args)
