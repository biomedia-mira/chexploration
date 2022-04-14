from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm

white = "White"
asian = "Asian"
black = "Black"


def get_boostrap_ci_for_split_sex_experiment(
    targets_sex: np.ndarray,
    predictions: np.ndarray,
    race: np.ndarray,
    n_bootstrap: int = 2000,
    level: float = 0.95,
):
    """
    Get all CIs for FPR/TPR/Youden/AUC per subgroup for SPLIT - sex experiment
    """
    n_samples = targets_sex.shape[0]

    all_fpr, all_tpr, all_roc_auc, all_youden = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    for b in tqdm(range(n_bootstrap + 1)):

        # GET BOOTSTRAP SAMPLE
        # At the first iteration, store the sample estimate without resampling.
        idx = (
            resample(np.arange(n_samples), stratify=targets_sex)
            if b > 0
            else np.arange(n_samples)
        )

        sample_sex, sample_pred, sample_race = (
            targets_sex[idx],
            predictions[idx],
            race[idx],
        )

        for r in [white, asian, black]:
            fpr, tpr, _ = roc_curve(
                sample_sex[sample_race == r], sample_pred[sample_race == r, 1]
            )
            all_roc_auc[r].append(auc(fpr, tpr))
            youden = tpr - fpr
            opt_youden_idx = np.argmax(youden)
            all_fpr[r].append(fpr[opt_youden_idx])
            all_tpr[r].append(tpr[opt_youden_idx])
            all_youden[r].append(youden[opt_youden_idx])

    def _get_pretty_string_from_bootstrap_estimates(boostrap_estimates: np.ndarray):
        alpha = (1 - level) / 2
        return f"{boostrap_estimates[0]: .2f} ({np.quantile(boostrap_estimates[1:], alpha):.2f}-{np.quantile(boostrap_estimates[1:], 1 - alpha):.2f})"

    return {
        "AUC": {
            asian: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[asian]),
            black: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[black]),
            white: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[white]),
        },
        "TPR": {
            white: _get_pretty_string_from_bootstrap_estimates(all_tpr[white]),
            black: _get_pretty_string_from_bootstrap_estimates(all_tpr[black]),
            asian: _get_pretty_string_from_bootstrap_estimates(all_tpr[asian]),
        },
        "FPR": {
            black: _get_pretty_string_from_bootstrap_estimates(all_fpr[black]),
            white: _get_pretty_string_from_bootstrap_estimates(all_fpr[white]),
            asian: _get_pretty_string_from_bootstrap_estimates(all_fpr[asian]),
        },
        "Youden's Index": {
            black: _get_pretty_string_from_bootstrap_estimates(all_youden[black]),
            white: _get_pretty_string_from_bootstrap_estimates(all_youden[white]),
            asian: _get_pretty_string_from_bootstrap_estimates(all_youden[asian]),
        },
    }


if __name__ == "__main__":

    # PATH TO PREDICTION AND DATA CHARACTERISTICS FILE
    cnn_pred = pd.read_csv(
        "../prediction/chexpert/sex/densenet-disease-all/predictions.test.csv"
    )
    data_characteristics = pd.read_csv("../datafiles/chexpert/chexpert.sample.test.csv")

    # PARAMETERS FOR CI
    n_bootstrap = 2000
    ci_level = 0.95

    # GET RESULTS
    results = get_boostrap_ci_for_split_sex_experiment(
        targets_sex=cnn_pred.target.values,
        predictions=cnn_pred[["class_0", "class_1"]].values,
        race=data_characteristics.race.values,
        n_bootstrap=n_bootstrap,
    )

    columns_as_in_manuscript = [white, asian, black]
    res_df = pd.DataFrame.from_dict(results, orient="index")[columns_as_in_manuscript]
    print(
        f"\nResults for SPLIT - SEX ({ci_level * 100:.0f}%-CI with {n_bootstrap} bootstrap samples)"
    )
    print(tabulate(res_df, headers=res_df.columns))
