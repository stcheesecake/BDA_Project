import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data")
    result_dir = os.path.join(data_dir, "kobert_results")

    parser.add_argument("--train_path", default=os.path.join(data_dir, "train.csv"))
    parser.add_argument("--test_path", default=os.path.join(data_dir, "test.csv"))
    parser.add_argument("--submission_path", default=os.path.join(data_dir, "sample_submission.csv"))
    parser.add_argument("--result_dir", default=result_dir)
    parser.add_argument("--kobert_path", default=str(Path(base_dir) / "models" / "kobert-base-v1"))
    parser.add_argument("--output_train", default=os.path.join(result_dir, "kobert_train.npy"))
    parser.add_argument("--output_test", default=os.path.join(result_dir, "kobert_test.npy"))
    parser.add_argument("--output_y", default=os.path.join(result_dir, "kobert_y.npy"))
    parser.add_argument("--skip_if_exists", action="store_true")
    return parser.parse_args()


def get_kobert_embedding(texts, kobert_path):
    kobert_path = str(Path(kobert_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(kobert_path, use_fast=False)
    model = AutoModel.from_pretrained(kobert_path)
    model.eval()

    embeddings = []
    for text in tqdm(texts, desc="🔍 KoBERT 임베딩 중"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.pooler_output.squeeze().numpy()
        embeddings.append(emb)
    return np.array(embeddings)


def preprocess_for_kobert(train_df, test_df):
    y = train_df['withdrawal']
    train_df = train_df.drop(columns=['withdrawal'])

    min_samples = int(len(train_df) * 0.1)
    low_sample_cols = train_df.columns[train_df.apply(lambda col: col.value_counts().iloc[0] < min_samples)]
    constant_cols = train_df.columns[train_df.nunique() == 1]
    drop_cols = set(['ID']) | set(low_sample_cols) | set(constant_cols)

    train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
    test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].fillna("해당 없음").astype(str)
            test_df[col] = test_df[col].fillna("해당 없음").astype(str)
        else:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)

    text_columns = train_df.select_dtypes(include='object').columns.tolist()
    numeric_columns = train_df.select_dtypes(include=['int', 'float']).columns.tolist()

    train_df['text_input'] = train_df[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    test_df['text_input'] = test_df[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    return train_df, test_df, text_columns, numeric_columns, y


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if args.skip_if_exists and os.path.exists(args.output_train) and os.path.exists(args.output_test):
        print("✅ KoBERT 임베딩이 이미 존재합니다. 생략합니다.")
        return

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    train_df, test_df, text_columns, numeric_columns, y = preprocess_for_kobert(train_df, test_df)

    train_embed = get_kobert_embedding(train_df['text_input'].tolist(), args.kobert_path)
    test_embed = get_kobert_embedding(test_df['text_input'].tolist(), args.kobert_path)

    np.save(args.output_train, train_embed)
    np.save(args.output_test, test_embed)
    np.save(args.output_y, y.to_numpy())
    print("✅ KoBERT 임베딩 저장 완료")


if __name__ == "__main__":
    main()