import os
import argparse
import pandas as pd
import numpy as np
import random
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from datetime import datetime
import optuna
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "..", "..", "train_test_data")
    kobert_dir = os.path.join(data_dir, "kobert_embedding")
    result_dir = os.path.join(base_dir, "..", "..", "..", "data", "results")

    parser.add_argument("--train_path", default=os.path.join(data_dir,"train.csv"))
    parser.add_argument("--test_path", default=os.path.join(data_dir,"test.csv"))
    parser.add_argument("--submission_path", default=os.path.join(data_dir,"sample_submission.csv"))
    parser.add_argument("--result_dir", default=result_dir)

    parser.add_argument("--kobert_train", default=os.path.join(kobert_dir, "kobert_train.npy"))
    parser.add_argument("--kobert_test", default=os.path.join(kobert_dir, "kobert_test.npy"))
    parser.add_argument("--kobert_y", default=os.path.join(kobert_dir, "kobert_y.npy"))

    parser.add_argument("--PCA_dim_range", default="32:512:1")
    parser.add_argument("--weight_0_range", default="1.5:4.0:0.1")
    parser.add_argument("--weight_1", type=float, default=1.0)
    parser.add_argument("--percentile", type=float, default=0.1)

    parser.add_argument("--iterations_range", default="800:1000:100")
    parser.add_argument("--depth_range", default="3:10:1")
    parser.add_argument("--learning_rate_range", default="0.01:0.05:0.01")
    parser.add_argument("--loss_function_list", default="CrossEntropy,Logloss")
    parser.add_argument("--threshold_range", default="0.501:0.57:0.001")
    parser.add_argument("--l2_leaf_reg_range", default="1.0:3.0:0.1")
    parser.add_argument("--grow_policy_list", default="SymmetricTree,Depthwise")
    parser.add_argument("--bootstrap_type_list", default="Bayesian,Bernoulli")
    parser.add_argument("--early_stopping_rounds_range", default="50")

    parser.add_argument("--n_trials", type=int, default=5000)
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

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    result_path = os.path.join(args.result_dir, f"{timestamp}_bo_results.csv")

    set_random_seed(args.random_seed)

    PCA_dims = parse_range_string(args.PCA_dim_range, is_float=False)
    weight_0_list = parse_range_string(args.weight_0_range, is_float=True)
    iterations_list = parse_range_string(args.iterations_range)
    depth_list = parse_range_string(args.depth_range)
    lr_list = parse_range_string(args.learning_rate_range, is_float=True)
    threshold_list = parse_range_string(args.threshold_range, is_float=True)
    l2_leaf_reg_list = parse_range_string(args.l2_leaf_reg_range, is_float=True)
    loss_function_list = parse_list_string(args.loss_function_list)
    grow_policy_list = parse_list_string(args.grow_policy_list)
    bootstrap_type_list = parse_list_string(args.bootstrap_type_list)
    early_stop_list = parse_range_string(args.early_stopping_rounds_range)

    if args.kobert_train and args.kobert_test and args.kobert_y:
        X = np.load(args.kobert_train).astype(np.float32)
        X_test = np.load(args.kobert_test).astype(np.float32)
        y = np.load(args.kobert_y)
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
        X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, stratify=y, random_state=args.random_seed)
        X_test = test

    results = []

    def objective(trial):
        trial_data = {
            "TRIAL": trial.number,
            "kobert_train": args.kobert_train,
            "kobert_test": args.kobert_test,
            "kobert_y": args.kobert_y,
            "weight_1": args.weight_1,
            "percentile": args.percentile
        }

        params = {
            'iterations': trial.suggest_categorical('iterations', iterations_list),
            'depth': trial.suggest_categorical('depth', depth_list),
            'learning_rate': trial.suggest_categorical('learning_rate', lr_list),
            'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', l2_leaf_reg_list),
            'loss_function': trial.suggest_categorical('loss_function', loss_function_list),
            'grow_policy': trial.suggest_categorical('grow_policy', grow_policy_list),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', bootstrap_type_list),
            'early_stopping_rounds': trial.suggest_categorical('early_stopping_rounds', early_stop_list),
            'verbose': args.verbose,
            'random_seed': args.random_seed,
            'eval_metric': "F1",
            'task_type': "GPU",
            'thread_count': 1
        }

        if cat_features is not None:
            params['cat_features'] = cat_features

        if params['loss_function'] == 'Logloss':
            weight_0 = trial.suggest_categorical('weight_0', weight_0_list)
            params['class_weights'] = {0: weight_0, 1: args.weight_1}
            trial_data["weight_0"] = weight_0

        threshold = trial.suggest_categorical('threshold', threshold_list)
        PCA_dim = trial.suggest_categorical('PCA_dim', PCA_dims)
        trial_data["threshold"] = threshold
        trial_data["PCA_dim"] = PCA_dim

        for key in ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg', 'loss_function',
                    'grow_policy', 'bootstrap_type', 'early_stopping_rounds']:
            trial_data[key] = params[key]

        X_tr, X_vl, Xt = X_train, X_val, X_test
        if PCA_dim != -1 and cat_features is None:
            pca = PCA(n_components=PCA_dim)
            X_tr = pca.fit_transform(X_train)
            X_vl = pca.transform(X_val)
            Xt = pca.transform(X_test)

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_train, eval_set=(X_vl, y_val))

        val_proba = model.predict_proba(X_vl)[:, 1]
        val_pred = (val_proba >= threshold).astype(int)

        trial_data["f1_score"] = f1_score(y_val, val_pred)
        trial_data["precision"] = precision_score(y_val, val_pred)
        trial_data["recall"] = recall_score(y_val, val_pred)
        trial_data["accuracy_0"] = accuracy_score(y_val[y_val == 0], val_pred[y_val == 0])
        trial_data["accuracy_1"] = accuracy_score(y_val[y_val == 1], val_pred[y_val == 1])

        results.append(trial_data)
        pd.DataFrame(results).to_csv(result_path, index=False)

        return trial_data["f1_score"]

    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n🛑 Optimization manually interrupted.")
    finally:
        if results:
            best_result = max(results, key=lambda x: x['f1_score'])
            with open(os.path.join(args.result_dir, f"{timestamp}_bo_best_result.txt"), "w", encoding="utf-8") as f:
                for k, v in best_result.items():
                    if k not in {"verbose", "train_path", "test_path", "submission_path", "result_dir"}:
                        f.write(f"{k}: {v}\n")
            print(f"✅ Saved best_result.txt after interruption.")

if __name__ == "__main__":
    main()
