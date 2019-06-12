#! /usr/bin/env python3

import csv
import json
import logging
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from util.serialization import load_class
from util.shared import parse_args


def build_classification_report(estimator, Xy, classes):
    def __build_report(result_df, average=None, labels=classes, index=classes):
        return pd.DataFrame(precision_recall_fscore_support(result_df["y_gold"], result_df["y_pred"],
                                                            labels=labels, average=average,
                                                            beta=estimator.F_BETA),
                            index=["precision", "recall", "f-beta", "support"], columns=index).T

    score_df = pd.concat([__build_report(Xy), __build_report(Xy, average="weighted", index=["-"])], keys=["detailed", "overall"])
    score_df.loc[("overall", "-"), "support"] = Xy.shape[0]
    if estimator.name() == "SpecialtyClassifier":
        from extern.database import _get_specialty_tree
        specialty_df = _get_specialty_tree().set_index("soc_code")

        Xy = Xy.copy()
        Xy["soc_code__major"] = specialty_df.loc[Xy["y_gold"], "soc_code__major"].values
        detailed_df, overall_df = score_df.loc["detailed"], score_df.loc["overall"]
        concatenable = {}
        for major_soc in Xy["soc_code__major"].unique():
            major_df = Xy[Xy["soc_code__major"] == major_soc]
            concatenable[major_soc] = __build_report(major_df, average="weighted", labels=None, index=["-"])
            concatenable[major_soc].loc["-", "support"] = major_df.shape[0]
        major_df = pd.concat(concatenable.values(), keys=concatenable.keys(), sort=True).xs("-", axis=0, level=1)
        score_df = pd.concat([detailed_df, major_df, overall_df], keys=["detailed", "major", "overall"])
        score_df["name"] = specialty_df.loc[[ix[-1] for ix in score_df.index], "name"].values
    score_df["support"] = score_df["support"].astype(int)

    cols = ["f-beta", "precision", "recall", "support"]
    cols += [c for c in score_df if c not in cols]
    score_df = score_df[cols]
    return score_df, score_df.loc[("overall", "-"), "f-beta"]

def build_confusion_matrix(estimator, Xy, classes, fbeta_score, ylim=[0, 1.000],
                           now=pd.Timestamp.now(tz="US/Pacific"),
                           confusion_min=0.001, confusion_ct=3):
    cnf_matrix = pd.DataFrame(confusion_matrix(Xy["y_gold"], Xy["y_pred"], labels=classes),
                              columns=classes, index=classes)
    cnf_matrix = cnf_matrix.divide(cnf_matrix.sum(axis=1), axis=0)

    labels = classes
    if estimator.name() == "SpecialtyClassifier":
        from extern.database import _get_specialty_tree
        labels_s = _get_specialty_tree().set_index("soc_code").loc[labels, "soc_code__major"]
        labels_s[labels_s.duplicated(keep="first")] = ""
        labels = labels_s.tolist()

    fig = sns.heatmap(cnf_matrix.fillna(0.000), cbar=False, cmap="inferno", vmin=ylim[0], vmax=ylim[1],
                      xticklabels=labels, yticklabels=labels)
    plt.title("{:.1%} {:} Confusion Matrix [{:%Y-%m-%d %H:%M}]".format(fbeta_score, estimator.name(), now))
    plt.tight_layout()

    cnf_df = cnf_matrix.reset_index() \
                       .melt(id_vars="index") \
                       .query("value >= %f" % confusion_min) \
                       .sort_values("value", ascending=False) \
                       .groupby("index") \
                       .head(confusion_ct)
    cnf_s = cnf_df.groupby("index")["variable"].apply(list)

    __remove = lambda x, y: [i for i in x if i != y]
    cnf_s = pd.Series([__remove(item, name) for name, item in cnf_s.iteritems()], index=cnf_s.index)

    __safe_len = lambda x: len(x) if isinstance(x, list) else 0
    cnf_s[cnf_s.apply(__safe_len) < 1] = np.nan
    return fig, cnf_s

def _subprocess_run(*args):
    cp = subprocess.run(args, capture_output=True)
    cp.check_returncode()
    return cp.stdout

def stash_for_posterity(name, now):
    stash = "{:%Y-%m-%d %H:%M}".format(now)
    stash_stdout = _subprocess_run("git", "stash", "save", "-u", "Saving for {:}: {:}".format(name, stash)).decode()
    if "nothing to commit, working tree clean" not in stash_stdout and \
       "No local changes to save" not in stash_stdout:
        apply_stdout = _subprocess_run("git", "stash", "apply")

    hash = _subprocess_run("git", "log", "--oneline", "-1").split()[0].decode()
    return {
        "commit": hash,
        "stash": stash,
    }

if __name__ == "__main__":
    args = parse_args("Estimate confusion matrix for any BaseEstimator-based model.", [
        {"name_or_flags": "--no-force", "dest": "force", "action": "store_false", "help": "don't rebuild model from scratch" },
        {"name_or_flags": "model_cls", "type": load_class, "help": "load this class" },
    ])
    details = stash_for_posterity(args.prog, args.started_at)

    def _output(ext):
        return os.path.join(args.output, "{:%H-%M}.{:}".format(args.started_at, ext))
    json_kwargs = dict(indent=2, separators=[',', ': '])
    json.dump(details, open(_output("json"), "w"), **json_kwargs)

    estimator = args.model_cls.get(force=args.force, fit=False, use_s3=False)  # disabling S3 upload for big models

    # FIXME: Fix logging with n_jobs != 1:
    # https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
    X, y_gold = estimator.goldset()
    y_pred = estimator._cv_predict(estimator, X, y_gold, random_state=args.seed)

    Xy = X.copy()
    Xy["y_gold"] = y_gold.apply(estimator._humanize_class)
    Xy["y_pred"] = pd.Series(y_pred, index=X.index).apply(estimator._humanize_class)
    classes = sorted([estimator._humanize_class(cls) for cls in estimator._get_scoreable_labels()])

    score_df, fbeta_score = build_classification_report(estimator, Xy, classes)
    fig, cnf_s = build_confusion_matrix(estimator, Xy, classes, fbeta_score, now=args.started_at)

    if isinstance(score_df.index, pd.MultiIndex):
        score_df["confusion"] = cnf_s.loc[[ix[-1] for ix in score_df.index]].values
    else:
        score_df["confusion"] = cnf_s.loc[score_df.index].values
    details["fbeta"] = {
        "score": fbeta_score,
        "beta": estimator.F_BETA,
    }

    json.dump(details, open(_output("json"), "w"), **json_kwargs)  # re-dump score details
    score_df.round(3).to_csv(_output("score.tsv"), sep="\t")

    if estimator.name() == "SpecialtyClassifier":
        Xy = Xy[[c for c in Xy if c != "raw_text"] + ["raw_text"]]
        Xy["raw_text"] = "u" + Xy["raw_text"].apply(json.dumps)
    Xy.to_csv(_output("raw.tsv"), sep="\t", quoting=csv.QUOTE_NONE)
    plt.savefig(_output("png"))

    __pct = lambda x: "{:0.0%}".format(x) if pd.notnull(x) and x > 0 else ""
    formatters = { col: __pct for col in score_df if col not in ["support", "name", "confusion"] }
    logging.info("Calculated overall %0.1f%% F-%0.3f score:\n%s", fbeta_score * 100, estimator.F_BETA,
                 score_df.to_string(na_rep="", formatters=formatters))
    plt.show()

    # pd.DataFrame(estimator.get_params().values(), index=estimator.get_params().keys()).loc[["voter__alpha", "voter__title__absolute_min", "voter__title__alpha", "voter__title__beta"]]

    # from hlda.sampler import HierarchicalLDA
    # corpus = X["_title_tokens"].tolist()
    # vocab_df = pd.DataFrame(Counter(flatten(corpus)).most_common(), columns=["word", "count"]).set_index("word")
    # vocab_df["ix"] = range(vocab_df.shape[0])
    # new_corpus = [vocab_df.loc[doc, "ix"].values.tolist() for doc in corpus]
    # tfm = HierarchicalLDA(new_corpus, vocab_df.index.tolist(), alpha=10.000, gamma=1.000, eta=0.100, num_levels=3)
    # tfm.estimate(1000, display_topics=1000, n_words=3, with_weights=True)
    import pdb; pdb.set_trace()
