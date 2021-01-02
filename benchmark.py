from itertools import product
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from pathlib import Path
from bench_utils import DataInfo
from bench_utils import fetch_openml_and_clean
from bench_utils import get_estimator
from bench_utils import write_results
from bench_utils import get_results_path
from sklearn.model_selection import cross_validate

from category_encoders import JamesSteinEncoder
from sk_encoder_cv import NestedEncoderCV
from sk_encoder_cv import TargetRegressorEncoder
from sk_encoder_cv import TargetRegressorEncoderCV


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
    "JamesSteinEncoder": JamesSteinEncoder(),
    "JamesSteinEncoderCV": NestedEncoderCV(JamesSteinEncoder()),
}


def run_single_benchmark(data_str, encoder_str, cv, n_jobs, write_result, force):
    print(f"running benchmark for {data_str} and {encoder_str}")
    data_info = DATA_INFOS[data_str]
    encoder = ENCODERS[encoder_str]

    results_path = get_results_path(RESULTS_DIR, data_info, encoder_str)
    if results_path.exists() and not force:
        print(
            f"benchmark for {data_str} and {encoder_str} exists pass --force to rerun"
        )
        return

    X, y = fetch_openml_and_clean(data_info=data_info)
    estimator = get_estimator(encoder, data_info)

    if data_info.is_classification:
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
    write_results(results, RESULTS_DIR, data_info, encoder_str, cv, write_result)


def _run_single_benchmark(args):
    run_single_benchmark(
        data_str=args.dataset,
        encoder_str=args.encoder,
        cv=args.cv,
        n_jobs=args.n_jobs,
        write_result=not args.no_write,
        force=args.force,
    )


def _run_all_benchmark(args):
    print("running all benchmarks")
    for data_str, encoder_str in product(DATA_INFOS, ENCODERS):
        run_single_benchmark(
            data_str, encoder_str, args.cv, args.n_jobs, not args.no_write, args.force
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
    single_parser.add_argument("encoder", choices=ENCODERS)

    single_parser.set_defaults(func=_run_single_benchmark)

    run_all_parser = subparsers.add_parser("all", help="Run all benchmarks")
    run_all_parser.set_defaults(func=_run_all_benchmark)

    args = parser.parse_args()
    args.func(args)
