import os
import argparse
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import itertools
from datetime import datetime
import sys
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    data_dir = os.path.join(base_dir, "..", "..", "data")
    result_dir = os.path.join(data_dir, "results")

    parser.add_argument("--train_path", default=os.path.join(data_dir, "train.csv"))
    parser.add_argument("--test_path", default=os.path.join(data_dir, "test.csv"))
    parser.add_argument("--submission_path", default=os.path.join(data_dir, "sample_submission.csv"))
    parser.add_argument("--result_dir", default=result_dir)

    parser.add_argument("--kobert_train", default="C:/Users/ilumi/BDA_Project/data/kobert_results/kobert_train.npy")
    parser.add_argument("--kobert_test", default="C:/Users/ilumi/BDA_Project/data/kobert_results/kobert_test.npy")
    parser.add_argument("--kobert_y", default="C:/Users/ilumi/BDA_Project/data/kobert_results/kobert_y.npy")

    parser.add_argument("--PCA_dim", default="32")

    parser.add_argument("--weight_0_range", default="2")
    parser.add_argument("--weight_1", type=float, default=1.0)
    parser.add_argument("--percentile", type=float, default=0.5)

    parser.add_argument("--iterations_range", default="1000")
    parser.add_argument("--depth_range", default="9")
    parser.add_argument("--learning_rate_range", default="0.005")
    parser.add_argument("--loss_function_list", default="CrossEntropy")
    parser.add_argument("--threshold_range", default="0.5")

    parser.add_argument("--l2_leaf_reg_range", default="3.0")
    parser.add_argument("--grow_policy_list", default="SymmetricTree")
    parser.add_argument("--bootstrap_type_list", default="Bayesian")

    parser.add_argument("--eval_metric", default="F1")
    parser.add_argument("--early_stopping_rounds", default="50")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", type=bool, default=False)

    return parser.parse_args()

def parse_range_string(s, is_float=False):
    if ":" in s:
        parts = list(map(float if is_float else int, s.split(":")))
        if len(parts) != 3:
            raise ValueError(f"Invalid range format: {s}")
        start, end, step = parts
        return list(np.round(np.arange(start, end + 1e-9, step), 5)) if is_float else list(range(int(start), int(end + 1), int(step)))
    else:
        return [float(s)] if is_float else [int(s)]

def parse_list_string(s):
    return s.split(",")

def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    PCA_dims = parse_range_string(args.PCA_dim, is_float=False)
    PCA_dim = PCA_dims[0] if len(PCA_dims) == 1 else -1

    if args.kobert_train and args.kobert_test and args.kobert_y:
        X = np.load(args.kobert_train)
        X_test = np.load(args.kobert_test)
        y = np.load(args.kobert_y)

        if PCA_dim != -1:
            pca = PCA(n_components=PCA_dim)
            X = pca.fit_transform(X)
            X_test = pca.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.random_seed)
        cat_features = None
    else:
        train = pd.read_csv(args.train_path)
        test = pd.read_csv(args.test_path)
        submission = pd.read_csv(args.submission_path)

        y = train['withdrawal']
        train = train.drop(columns=['withdrawal'])

        drop_cols = ['ID']
        drop_cols.extend(train.nunique()[train.nunique() == 1].index.tolist())
        valid_counts = train.notnull().sum()
        threshold_count = valid_counts.quantile(args.percentile)
        drop_cols.extend(valid_counts[valid_counts < threshold_count].index.tolist())
        drop_cols = list(set(drop_cols))

        train = train.drop(columns=drop_cols)
        test = test.drop(columns=drop_cols)

        train = train.astype(str).fillna('missing')
        test = test.astype(str).fillna('missing')
        cat_features = train.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            train, y, test_size=0.2, stratify=y, random_state=args.random_seed
        )

    if PCA_dim != -1 and cat_features is None:
        pca = PCA(n_components=PCA_dim)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

    iterations_list = parse_range_string(args.iterations_range)
    depth_list = parse_range_string(args.depth_range)
    learning_rate_list = parse_range_string(args.learning_rate_range, is_float=True)
    weight_0_list = parse_range_string(args.weight_0_range, is_float=True)
    loss_function_list = parse_list_string(args.loss_function_list)
    threshold_list = parse_range_string(args.threshold_range, is_float=True)
    l2_leaf_reg_list = parse_range_string(args.l2_leaf_reg_range, is_float=True)
    grow_policy_list = parse_list_string(args.grow_policy_list)
    bootstrap_type_list = parse_list_string(args.bootstrap_type_list)
    early_stopping_list = parse_range_string(args.early_stopping_rounds)

    param_combinations = list(itertools.product(
        iterations_list,
        depth_list,
        learning_rate_list,
        weight_0_list,
        loss_function_list,
        threshold_list,
        l2_leaf_reg_list,
        grow_policy_list,
        bootstrap_type_list,
        early_stopping_list
    ))

    best_f1 = -1
    best_model = None
    best_params = {}
    best_val_pred = None
    results = []

    for iter_, depth_, lr_, w0, loss_func, threshold, l2_, grow_pol, boot_type, early_stop in tqdm(param_combinations, desc="🔍 Grid Searching"):
        model_params = {
            'iterations': int(iter_),
            'learning_rate': lr_,
            'depth': int(depth_),
            'loss_function': loss_func,
            'eval_metric': args.eval_metric,
            'l2_leaf_reg': l2_,
            'bootstrap_type': boot_type,
            'grow_policy': grow_pol,
            'early_stopping_rounds': early_stop,
            'verbose': args.verbose,
            'random_seed': args.random_seed,
            'task_type': "GPU"
        }
        if cat_features is not None:
            model_params['cat_features'] = cat_features
        if loss_func == 'Logloss':
            model_params['class_weights'] = {0: w0, 1: args.weight_1}

        model = CatBoostClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= threshold).astype(int)

        f1 = f1_score(y_val, val_pred)
        precision = precision_score(y_val, val_pred)
        recall = recall_score(y_val, val_pred)
        acc_total = accuracy_score(y_val, val_pred)
        acc_0 = accuracy_score(y_val[y_val == 0], val_pred[y_val == 0])
        acc_1 = accuracy_score(y_val[y_val == 1], val_pred[y_val == 1])

        results.append({
            "iterations": iter_,
            "depth": depth_,
            "learning_rate": lr_,
            "weight_0": w0,
            "weight_1": args.weight_1,
            "loss_function": loss_func,
            "threshold": threshold,
            "l2_leaf_reg": l2_,
            "grow_policy": grow_pol,
            "bootstrap_type": boot_type,
            "early_stopping_rounds": early_stop,
            "PCA_dim": PCA_dim,
            "percentile": args.percentile,
            "eval_metric": args.eval_metric,
            "random_seed": args.random_seed,
            "verbose": args.verbose,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "val_acc": acc_total,
            "acc_class_0": acc_0,
            "acc_class_1": acc_1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_val_pred = val_pred
            best_params = results[-1]

    pd.DataFrame(results).to_csv(os.path.join(args.result_dir, f"{timestamp}_grid_search_results.csv"), index=False)

    with open(os.path.join(args.result_dir, f"{timestamp}_best_result.txt"), "w", encoding="utf-8") as f:
        f.write("Best Params:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest F1 Score: {best_f1:.4f}\n")
        f.write(f"Val Accuracy: {accuracy_score(y_val, best_val_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_val, best_val_pred):.4f}\n")
        f.write(f"Recall: {recall_score(y_val, best_val_pred):.4f}\n")
        f.write(f"Class 0 Accuracy: {accuracy_score(y_val[y_val == 0], best_val_pred[y_val == 0]):.4f}\n")
        f.write(f"Class 1 Accuracy: {accuracy_score(y_val[y_val == 1], best_val_pred[y_val == 1]):.4f}\n")

    X_test = np.load(args.kobert_test) if args.kobert_test else test
    if PCA_dim != -1 and cat_features is None:
        X_test = pca.transform(X_test)

    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_params['threshold']).astype(int)

    submission = pd.read_csv(args.submission_path)
    submission['withdrawal'] = test_pred
    submission.to_csv(os.path.join(args.result_dir, f"{timestamp}_submission_catboost.csv"), index=False)

if __name__ == "__main__":
    main()
