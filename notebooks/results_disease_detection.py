from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import auc, recall_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm

target_fpr = 0.2


white = "White"
asian = "Asian"
black = "Black"
male = "Male"
female = "Female"

labels = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def get_boostrap_ci_for_full_experiment(
    targets: np.ndarray,
    predictions: np.ndarray,
    race: np.ndarray,
    sex: np.ndarray,
    n_bootstrap: int = 2000,
    level: float = 0.95,
):
    """
    Get all CIs for FPR/TPR/Youden/AUC per subgroup for a global threshold with target fpr of 0.2
    """
    n_samples = targets.shape[0]

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
            resample(np.arange(n_samples), stratify=targets)
            if b > 0
            else np.arange(n_samples)
        )

        sample_target, sample_pred = targets[idx], predictions[idx]
        sample_race, sample_sex = race[idx], sex[idx]

        fpr, tpr, thres = roc_curve(sample_target, sample_pred)
        all_roc_auc["all"].append(auc(fpr, tpr))

        # Computing global threshold
        idx_target_fpr_threshold = np.argmin(np.abs(fpr - target_fpr))
        op = thres[idx_target_fpr_threshold]
        all_fpr["all"].append(fpr[idx_target_fpr_threshold])
        all_tpr["all"].append(tpr[idx_target_fpr_threshold])
        all_youden["all"].append(
            (tpr[idx_target_fpr_threshold] - fpr[idx_target_fpr_threshold])
        )

        # Getting race subbroups results
        for r in [white, asian, black]:
            targets_r, preds_r = (
                sample_target[sample_race == r],
                sample_pred[sample_race == r],
            )
            all_roc_auc[r].append(roc_auc_score(targets_r, preds_r))
            all_fpr[r].append(1 - recall_score(targets_r, preds_r >= op, pos_label=0))
            all_tpr[r].append(recall_score(targets_r, preds_r >= op, pos_label=1))
            all_youden[r].append(all_tpr[r][-1] - all_fpr[r][-1])

        # Getting sex subgroup results
        for s in [male, female]:
            targets_s, preds_s = (
                sample_target[sample_sex == s],
                sample_pred[sample_sex == s],
            )
            all_roc_auc[s].append(roc_auc_score(targets_s, preds_s))
            all_fpr[s].append(1 - recall_score(targets_s, preds_s >= op, pos_label=0))
            all_tpr[s].append(recall_score(targets_s, preds_s >= op, pos_label=1))
            all_youden[s].append(all_tpr[s][-1] - all_fpr[s][-1])

    def _get_pretty_string_from_bootstrap_estimates(boostrap_estimates: np.ndarray):
        alpha = (1 - level) / 2
        return f"{boostrap_estimates[0]: .2f} ({np.quantile(boostrap_estimates[1:], alpha):.2f}-{np.quantile(boostrap_estimates[1:], 1 - alpha):.2f})"

    return {
        "AUC": {
            asian: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[asian]),
            black: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[black]),
            white: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[white]),
            male: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[male]),
            female: _get_pretty_string_from_bootstrap_estimates(all_roc_auc[female]),
            "all": _get_pretty_string_from_bootstrap_estimates(all_roc_auc["all"]),
        },
        "TPR": {
            white: _get_pretty_string_from_bootstrap_estimates(all_tpr[white]),
            black: _get_pretty_string_from_bootstrap_estimates(all_tpr[black]),
            asian: _get_pretty_string_from_bootstrap_estimates(all_tpr[asian]),
            male: _get_pretty_string_from_bootstrap_estimates(all_tpr[male]),
            female: _get_pretty_string_from_bootstrap_estimates(all_tpr[female]),
            "all": _get_pretty_string_from_bootstrap_estimates(all_tpr["all"]),
        },
        "FPR": {
            black: _get_pretty_string_from_bootstrap_estimates(all_fpr[black]),
            white: _get_pretty_string_from_bootstrap_estimates(all_fpr[white]),
            asian: _get_pretty_string_from_bootstrap_estimates(all_fpr[asian]),
            male: _get_pretty_string_from_bootstrap_estimates(all_fpr[male]),
            female: _get_pretty_string_from_bootstrap_estimates(all_fpr[female]),
            "all": _get_pretty_string_from_bootstrap_estimates(all_fpr["all"]),
        },
        "Youden's Index": {
            black: _get_pretty_string_from_bootstrap_estimates(all_youden[black]),
            white: _get_pretty_string_from_bootstrap_estimates(all_youden[white]),
            asian: _get_pretty_string_from_bootstrap_estimates(all_youden[asian]),
            male: _get_pretty_string_from_bootstrap_estimates(all_youden[male]),
            female: _get_pretty_string_from_bootstrap_estimates(all_youden[female]),
            "all": _get_pretty_string_from_bootstrap_estimates(all_youden["all"]),
        },
    }


if __name__ == "__main__":

    # PATH TO PREDICTION AND DATA CHARACTERISTICS FILE
    cnn_pred = pd.read_csv(
        "../prediction/chexpert/disease/densenet-all/predictions.test.csv"
    )
    data_characteristics = pd.read_csv("../datafiles/chexpert/chexpert.sample.test.csv")

    # PARAMETERS FOR CI
    n_bootstrap = 2000
    ci_level = 0.95

    # GET RESULTS
    for label in [0, 10]:
        preds = cnn_pred["class_" + str(label)]
        targets = np.array(cnn_pred["target_" + str(label)])
        race = data_characteristics.race.values
        sex = data_characteristics.sex.values

        results = get_boostrap_ci_for_full_experiment(
            targets=targets,
            predictions=preds,
            race=race,
            sex=sex,
            n_bootstrap=n_bootstrap,
        )

        columns_as_in_manuscript = [white, asian, black, female, male, "all"]
        res_df = pd.DataFrame.from_dict(results, orient="index")[
            columns_as_in_manuscript
        ]
        print(
            f"\nResults for: {labels[label].upper()} ({ci_level * 100:.0f}%-CI with {n_bootstrap} bootstrap samples)"
        )
        print(tabulate(res_df, headers=res_df.columns))
