"""
BDA 학습자 수료 예측 EDA 스크립트
────────────────────────────────────────
1) train.csv 로드
2) 열별 유효·결측·고유 개수, dtype, 샘플 값 추출
3) 요약표(eda_summary) 저장
4) 상위 100개 Feature 유효 샘플 수 Bar 차트
5) 열별 dtype & 샘플 값 콘솔 출력
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
pd.set_option("display.max_colwidth", None)

# ── 경로 설정 ────────────────────────────────────────────────
DATA_DIR   = r"C:/Users/ilumi/BDA_Project/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
EDA_DIR    = os.path.join(DATA_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# ── 1) 데이터 불러오기 ───────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)

# ── 2) 열별 지표 계산 ────────────────────────────────────────
total_rows   = len(train)
valid_counts = train.notnull().sum()
null_counts  = train.isnull().sum()
data_types   = train.dtypes
unique_counts = train.nunique()
# 처음으로 널이 아닌 값 하나씩 가져오기
sample_values = train.apply(lambda s: next((v for v in s if pd.notna(v)), None))

eda_summary = pd.DataFrame({
    "Sample Count (Valid)": valid_counts,
    "Null Count": null_counts,
    "Data Type": data_types,
    "Unique Count": unique_counts,
    "Sample Value": sample_values
}).sort_values("Sample Count (Valid)", ascending=False)

print(f"총 샘플 개수: {total_rows}\n")
print("EDA 요약표:")
print(eda_summary)

# ── 3) 요약표 저장 ──────────────────────────────────────────
SUMMARY_PATH = os.path.join(EDA_DIR, "eda_summary.csv")
eda_summary.to_csv(SUMMARY_PATH, encoding="utf-8-sig")
print(f"\n✅ 요약표 저장 완료 → {SUMMARY_PATH}")

# ── 4) Bar 차트(유효 샘플 수 Top 100) ───────────────────────
top_valid = eda_summary["Sample Count (Valid)"].sort_values(ascending=True).tail(100)

plt.figure(figsize=(28, 16))
colors = cm.get_cmap("tab20", len(top_valid))(np.linspace(0, 1, len(top_valid)))  # 수정

top_valid.plot(kind="barh", color=colors, edgecolor="black")
plt.title(f"Features by Valid Sample Count (Total rows: {total_rows})")
plt.xlabel("Sample Count")
plt.tight_layout()

for i, value in enumerate(top_valid):
    plt.text(value + 5, i, f"{value}", va="center", fontsize=9)

plt.show()

# ── 5) 열별 dtype & 샘플 값 콘솔 출력 ────────────────────────
print("\n열별 dtype 및 샘플 값:")
for col in train.columns:
    dtype  = train[col].dtype
    sample = train[col].dropna().iloc[0] if not train[col].dropna().empty else None
    print(f"{col:<25} | dtype: {str(dtype):<10} | sample: {sample}")

# ── 6) dtype 및 샘플 값 포함한 EDA 상세표 저장 ────────────────────────

eda_detail = pd.DataFrame({
    "Feature": train.columns,
    "Data Type": train.dtypes.astype(str).values,
    "Valid Count": train.notnull().sum().values,
    "Unique Count": train.nunique().values,
    "Sample Value": train.apply(lambda s: next((v for v in s if pd.notna(v)), None)).values
})
eda_detail["Sample Value"] = eda_detail["Sample Value"].astype(str).str.replace(r'\s+', ' ', regex=True)

# 저장 경로
DETAIL_PATH = os.path.join(EDA_DIR, "eda_detail.tsv")

# 탭 구분자로 저장하되 확장자는 .csv
eda_detail.to_csv(DETAIL_PATH, sep='\t', index=False, encoding='utf-8-sig')
print(f"\n📁 dtype + sample 포함 요약표 저장 완료 → {DETAIL_PATH}")

print(train['previous_class_7'].dropna().unique())