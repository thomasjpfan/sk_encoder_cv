"""Used to generate README with the results of runs."""
from pathlib import Path
from matplotlib import figure
import numpy as np
import pandas as pd
from benchmark import DATASET_NAMES

RESULTS_PATH = Path("results")
FIGURES_PATH = Path("figures")
README_PATH = Path("README.md")

FIGURES_PNG_PATHS = [FIGURES_PATH / f"{name}.png" for name in DATASET_NAMES]
MD_DATASET_COLUMNS = [
    "data_name",
    "categorical features",
    "n_features",
    "n_samples",
    "is_classification",
    "openml_url",
]

# These datasts have very low performance for drop when compared to
# all the other encoders
DATASET_TO_REMOVE_DROP = {"adult", "Allstate_Claims_Severity"}


def plot_metric_for_name(
    data_name, metric_name, results_df, ax=None, remove_drop=False
):
    if ax is None:
        fig = figure.Figure()
        ax = fig.subplots()
    results_data_name = results_df[results_df["data_name"] == data_name]

    info_first = results_data_name.iloc[0]
    data_name = info_first["data_name"]
    results_data_name_sorted = results_data_name.sort_values(f"test_{metric_name}_mean")
    null_encoders = ~results_data_name_sorted[f"test_{metric_name}_mean"].isna()
    if remove_drop:
        null_encoders &= results_data_name_sorted["encoder"] != "drop"
    y_values = np.arange(np.sum(null_encoders))

    ax.errorbar(
        results_data_name_sorted.loc[null_encoders, f"test_{metric_name}_mean"],
        y_values,
        xerr=results_data_name_sorted.loc[null_encoders, f"test_{metric_name}_std"],
        ls="",
        marker="o",
    )
    ax.set_yticks(y_values)
    ax.set_yticklabels(results_data_name_sorted.loc[null_encoders, "encoder"])
    ax.set_title(f"{data_name}: {metric_name}")


def plot_all_metrics(data_name, results_df):
    results_data_name = results_df[results_df["data_name"] == data_name]
    info_first = results_data_name.iloc[0]

    non_null_names = info_first.notnull()
    test_names = info_first.index.str.startswith("test")
    score_names = info_first.index[non_null_names & test_names]
    score_means_names = score_names[score_names.str.endswith("_mean")]

    metric_names = [name[5:-5] for name in score_means_names]

    fig = figure.Figure(figsize=(20, 6), constrained_layout=True)
    axes = fig.subplots(1, len(metric_names))

    for metric_name, ax in zip(metric_names, axes.flatten()):
        remove_drop = data_name in DATASET_TO_REMOVE_DROP
        plot_metric_for_name(
            data_name, metric_name, results_df, ax=ax, remove_drop=remove_drop
        )

    return fig


results_dir = Path("results")
results_paths = results_dir.glob("*.csv")

results = []
for path in results_paths:
    results.append(pd.read_csv(path))
results_df = pd.concat(results, axis=0)

for dataset_name, fig_path in zip(DATASET_NAMES, FIGURES_PNG_PATHS):
    print(f"Generating plot for {dataset_name}")
    fig = plot_all_metrics(dataset_name, results_df)
    fig.savefig(fig_path)

md_names = [
    f"![{dataset}]({str(fig_path)})"
    for dataset, fig_path in zip(DATASET_NAMES, FIGURES_PNG_PATHS)
]
md_links_text = "\n".join(md_names)

md_dataset_meta = (
    results_df.drop_duplicates("data_name")[MD_DATASET_COLUMNS]
    .set_index("data_name")
    .loc[DATASET_NAMES]
    .reset_index()
)

results_text = md_dataset_meta.to_markdown(index=False)

MARKDOWN_RESULTS_TEMPLATE = f"""# Benchmarks for Target Encoder

Benchmarks for different forms of target encoder with 10-fold cross validation.

## Datasets

{results_text}

## Results

{md_links_text}

## How to run benchmarks

0. Clone repo:

```bash
git clone http://github.com/thomasjpfan/sk_encoder_cv
cd sk_encoder_cv
```

1. Create virtualenv and install `sk_encoder_cv`

```bash
conda create -n sk_encoder_cv python=3.8  # or use venv
conda activate sk_encoder_cv
python setup.py develop
```

2. Run single benchmarks:

```bash
python benchmark.py single adult
```

3. Or run all benchmarks

```bash
python benchmark.py all
```

The results will be written into the `results` directory.

4. Generate README report:

```bash
python generate_report.py
```
"""

with README_PATH.open("w") as f:
    f.write(MARKDOWN_RESULTS_TEMPLATE)
