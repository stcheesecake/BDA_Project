import os
import argparse
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
import lightgbm as lgb
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

    parser.add_argument("--PCA_dim_range", default="50:100:1")
    parser.add_argument("--weight_0_range", default="1.3:4.0:0.1")
    parser.add_argument("--weight_1", type=float, default=1.0)
    parser.add_argument("--percentile", type=float, default=0.1)

    parser.add_argument("--num_leaves_range", default="16:128:16")
    parser.add_argument("--learning_rate_range", default="0.01:0.05:0.01")
    parser.add_argument("--lambda_l1_range", default="0.0:1.0:0.1")
    parser.add_argument("--feature_fraction_range", default="0.6:1.0:0.1")
    parser.add_argument("--max_depth_range", default="3:15:2")
    parser.add_argument("--threshold_range", default="0.501:0.6:0.001")

    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", type=bool, default=False)

    return parser.parse_args()


def parse_range_string(s, is_float=False):
    if ":" in s:
        parts = list(map(float if is_float else int, s.split(":")))
        start, end, step = parts
        return list(np.round(np.arange(start, end + 1e-9, step), 5)) if is_float else list(range(int(start), int(end + 1), int(step)))
    else:
        return [float(s)] if is_float else [int(s)]


def main():
    args = parse_args()

    # ✅ 결정성 보장을 위한 시드 고정
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    result_path = os.path.join(args.result_dir, f"{timestamp}_bo_results_lgb.csv")

    # Range parsing
    PCA_dims = parse_range_string(args.PCA_dim_range, is_float=False)
    weight_0_list = parse_range_string(args.weight_0_range, is_float=True)
    num_leaves_list = parse_range_string(args.num_leaves_range)
    lr_list = parse_range_string(args.learning_rate_range, is_float=True)
    lambda_l1_list = parse_range_string(args.lambda_l1_range, is_float=True)
    feat_frac_list = parse_range_string(args.feature_fraction_range, is_float=True)
    max_depth_list = parse_range_string(args.max_depth_range)
    threshold_list = parse_range_string(args.threshold_range, is_float=True)

    # 데이터 로딩
    if args.kobert_train and args.kobert_test and args.kobert_y:
        X = np.load(args.kobert_train).astype(np.float32)
        X_test = np.load(args.kobert_test).astype(np.float32)
        y = np.load(args.kobert_y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.random_seed)
    else:
        train = pd.read_csv(args.train_path)
        test = pd.read_csv(args.test_path)
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
        train = train.fillna(0)
        test = test.fillna(0)
        X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, stratify=y, random_state=args.random_seed)
        X_test = test

    results = []

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'None',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': args.random_seed,
            'num_leaves': trial.suggest_categorical('num_leaves', num_leaves_list),
            'learning_rate': trial.suggest_categorical('learning_rate', lr_list),
            'lambda_l1': trial.suggest_categorical('lambda_l1', lambda_l1_list),
            'feature_fraction': trial.suggest_categorical('feature_fraction', feat_frac_list),
            'max_depth': trial.suggest_categorical('max_depth', max_depth_list),
        }

        weight_0 = trial.suggest_categorical('weight_0', weight_0_list)
        params['class_weight'] = {0: weight_0, 1: args.weight_1}

        threshold = trial.suggest_categorical('threshold', threshold_list)
        PCA_dim = trial.suggest_categorical('PCA_dim', PCA_dims)

        X_tr, X_vl, Xt = X_train, X_val, X_test
        if PCA_dim != -1:
            pca = PCA(n_components=PCA_dim)
            X_tr = pca.fit_transform(X_tr)
            X_vl = pca.transform(X_vl)
            Xt = pca.transform(Xt)

        # ✅ 결정성 보장 파라미터 추가
        model = lgb.LGBMClassifier(**params, deterministic=True, force_row_wise=True)
        model.fit(X_tr, y_train)

        val_pred = (model.predict_proba(X_vl)[:, 1] >= threshold).astype(int)

        f1 = f1_score(y_val, val_pred)
        precision = precision_score(y_val, val_pred)
        recall = recall_score(y_val, val_pred)
        acc_0 = accuracy_score(y_val[y_val == 0], val_pred[y_val == 0])
        acc_1 = accuracy_score(y_val[y_val == 1], val_pred[y_val == 1])

        trial_data = {
            'TRIAL': trial.number,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'acc_0': acc_0,
            'acc_1': acc_1,
            'weight_0': weight_0,
            'PCA_dim': PCA_dim,
            'threshold': threshold,
            **params
        }

        results.append(trial_data)
        pd.DataFrame(results).to_csv(result_path, index=False)
        return f1

    # ✅ Optuna 시드 고정
    sampler = TPESampler(seed=args.random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n🛑 Optimization manually interrupted.")
    finally:
        if results:
            best_result = max(results, key=lambda x: x['f1_score'])
            with open(os.path.join(args.result_dir, f"{timestamp}_bo_best_lgb.txt"), "w", encoding="utf-8") as f:
                for k, v in best_result.items():
                    f.write(f"{k}: {v}\n")
            print("\n✅ Best result saved.")


if __name__ == "__main__":
    main()