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
    data_dir = os.path.join(base_dir, "..", "..", "train_test_data")
    result_dir = os.path.join(data_dir, "kobert_embedding")

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
    kobert_path = Path(kobert_path).resolve()

    tokenizer = AutoTokenizer.from_pretrained(str(kobert_path), use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(str(kobert_path), local_files_only=True)
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

    # 1. 전체 컬럼 데이터 타입 출력
    print("\n📋 전체 컬럼 데이터 타입:")
    print(train_df.dtypes)

    # 2. 제거 조건 정의
    min_samples = int(len(train_df) * 0.1)
    low_sample_cols = train_df.columns[train_df.apply(lambda col: col.value_counts().iloc[0] < min_samples)]
    constant_cols = train_df.columns[train_df.nunique() == 1]
    drop_cols = set(['ID']) | set(low_sample_cols) | set(constant_cols)

    # 3. 삭제 컬럼 출력
    print(f"\n🗑️ 삭제된 컬럼들 ({len(drop_cols)}개):")
    for col in sorted(drop_cols):
        print(f" - {col}")

    # 컬럼 삭제
    train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
    test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # 4. 결측치 처리 및 타입 변환
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].fillna("해당 없음").astype(str)
            test_df[col] = test_df[col].fillna("해당 없음").astype(str)
        else:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)

    # 5. 텍스트 / 수치형 컬럼 분리
    text_columns = train_df.select_dtypes(include='object').columns.tolist()
    numeric_columns = train_df.select_dtypes(include=['int', 'float']).columns.tolist()

    print(f"\n✍️ KoBERT 텍스트 임베딩에 사용된 컬럼들 ({len(text_columns)}개):")
    for col in text_columns:
        print(f" - {col}")

    print(f"\n🔢 수치형 피처로 사용된 컬럼들 ({len(numeric_columns)}개):")
    for col in numeric_columns:
        print(f" - {col}")

    # 텍스트 통합 컬럼 생성
    train_df['text_input'] = train_df[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    test_df['text_input'] = test_df[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    return train_df, test_df, numeric_columns, y


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if args.skip_if_exists and os.path.exists(args.output_train) and os.path.exists(args.output_test):
        print("✅ KoBERT 임베딩이 이미 존재합니다. 생략합니다.")
        return

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    train_df, test_df, numeric_columns, y = preprocess_for_kobert(train_df, test_df)

    train_embed = get_kobert_embedding(train_df['text_input'].tolist(), args.kobert_path)
    test_embed = get_kobert_embedding(test_df['text_input'].tolist(), args.kobert_path)

    train_numeric = train_df[numeric_columns].to_numpy(dtype=np.float32)
    test_numeric = test_df[numeric_columns].to_numpy(dtype=np.float32)

    train_final = np.concatenate([train_embed, train_numeric], axis=1)
    test_final = np.concatenate([test_embed, test_numeric], axis=1)

    np.save(args.output_train, train_final)
    np.save(args.output_test, test_final)
    np.save(args.output_y, y.to_numpy())

    print("\n✅ 최종 데이터 형태 확인:")
    print(f" - train_embed.shape = {train_embed.shape}")
    print(f" - train_numeric.shape = {train_numeric.shape}")
    print(f" - train_final.shape = {train_final.shape}")

    print("✅ KoBERT 임베딩 + 수치형 피처 저장 완료")


if __name__ == "__main__":
    main()