#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

from util.serialization import load_class
from util.shared import parse_args


def plot_learning_curve(estimator, ylim=[0, 1.000]):
    plt.figure()

    title = "%s Learning Curve" % estimator.__class__.__name__
    plt.title(title)
    plt.xlabel("Training Examples (#)")
    plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)

    X, y = estimator.goldset()
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, n_jobs=1,
                                                            cv=estimator._cv_splitter(),
                                                            scoring=estimator.scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-Validation Score")

    plt.legend(loc="best")
    plt.show()

    # NOTE: Explicitly leaving in here for ML-related debugging script to
    # introspect learned model.
    import pdb; pdb.set_trace()
    return plt

if __name__ == "__main__":
    args = parse_args("Estimate learning curve for any BaseEstimator-based model.", [
        {"name_or_flags": "--no-force", "dest": "force", "action": "store_false", "help": "don't rebuild model from scratch" },
        {"name_or_flags": "model_cls", "type": load_class, "help": "load this class" },
    ])
    estimator = args.model_cls.get(force=args.force, use_s3=False)
    plot_learning_curve(estimator)
