import pandas as pd

# ─────────────── 파일 경로 설정 ───────────────
left_path  = r"C:/Users/ilumi/BDA_Project/data/eda/kor.csv"
right_path = r"C:/Users/ilumi/BDA_Project/data/eda/raw.csv"
out_path   = r"C:/Users/ilumi/BDA_Project/data/eda/kor_enriched.csv"

# ─────────────── CSV 읽기 함수 ───────────────
def robust_read(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"❌ 인코딩 실패: {path}", path, 0, 0, "invalid encoding")

# ─────────────── 데이터 불러오기 ───────────────
left_df  = robust_read(left_path)
right_df = robust_read(right_path)

# ─────────────── feature → feature_key, explain 분리 ───────────────
def split_feature(text: str):
    if ":" in text:
        left, right = text.split(":", 1)
        return left.strip(), right.strip()
    return text.strip(), ""

left_df[["feature_key", "explain"]] = left_df["feature"].apply(lambda x: pd.Series(split_feature(str(x))))

# ─────────────── 병합용 키 기준으로 수동 매핑 ───────────────
# 기준: right_df의 feature와 left_df의 feature_key가 동일하면 통계값 추출
right_df_indexed = right_df.set_index("feature")  # 인덱스를 feature로 지정

# 원하는 통계 컬럼 리스트
stats_cols = ["Sample Count (Valid)", "Null Count", "Data Type", "Unique Count"]

# left_df의 feature_key 기준으로 오른쪽에서 찾아서 붙이기
for col in stats_cols:
    left_df[col] = left_df["feature_key"].map(right_df_indexed[col] if col in right_df.columns else None)

# ─────────────── 최종 출력 컬럼 구성 ───────────────
final_df = left_df[["feature_key", "explain"] + stats_cols]
final_df.rename(columns={"feature_key": "feature"}, inplace=True)

# ─────────────── 저장 (탭 구분, 확장자는 .csv) ───────────────
final_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {out_path}")