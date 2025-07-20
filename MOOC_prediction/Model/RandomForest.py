import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --------------------- 전역 설정 ---------------------
TEST_SIZE = 0.2           # 검증 비율
RANDOM_SEED = 42          # 재현성
USE_TUNING = True         # 하이퍼파라미터 튜닝 여부 (True/False)

# ----------------- 1. 데이터 불러오기 -----------------
df = pd.read_csv("C:/Users/ilumi/BDA_Project/data/train.csv")

# -------------- 2. 타깃 분리 & 열 삭제 ----------------
y = df["withdrawal"].copy()
drop_cols = ["ID", "generation", "class4", "withdrawal"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# --------- 3. 결측률 90% 이상 열 제거 ------------------
high_null_cols = df.columns[df.isnull().mean() >= 0.9]
df.drop(columns=high_null_cols, inplace=True)

# --------------- 4. 결측치 “무응답” 보간 ---------------
df.fillna("무응답", inplace=True)

# --------- 5. 모든 컬럼 category → code 인코딩 ----------
for col in df.columns:
    df[col] = df[col].astype("category").cat.codes

y = y.astype("category").cat.codes  # 타깃 인코딩

# ------------- 6. 학습/검증 데이터 분할 ---------------
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# ------------- 7. RandomForest 모델 정의 ----------------
base_rf = RandomForestClassifier(
    random_state=RANDOM_SEED,
    class_weight="balanced",
    n_jobs=-1
)

# --------- 8. 하이퍼파라미터 튜닝 (GridSearchCV) --------
if USE_TUNING:
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 15],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt"]
    }

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        scoring="f1",  # 이진 분류에서 F1 사용
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    rf = grid_search.best_estimator_
    print("🎯 Best Params:", grid_search.best_params_)
else:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

# ------------- 9. 성능 평가 ---------------------------
y_pred = rf.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ------------- 10. Confusion Matrix & Heatmap ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0 (X)", "1 (O)"], yticklabels=["0 (X)", "1 (O)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# ------------- 11. Feature Importance ------------------
importances = rf.feature_importances_
feat_names  = df.columns
indices     = np.argsort(importances)[::-1]

plt.figure(figsize=(14, 6))
plt.title("RandomForest Feature Importances (Top 20)")
plt.bar(range(20), importances[indices][:20], align="center")
plt.xticks(range(20), [feat_names[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.show()

# ------------- 12. 클래스 분포 확인 --------------------
print("\n✅ 전체 클래스 분포:")
print(y.value_counts(normalize=True))

print("\n✅ 훈련셋 클래스 분포:")
print(y_train.value_counts(normalize=True))

print("\n✅ 검증셋 클래스 분포:")
print(y_test.value_counts(normalize=True))



# ----------------- 13. test.csv 예측 ------------------
# 경로 설정
TEST_PATH = "C:/Users/ilumi/BDA_Project/data/test.csv"
SUBMIT_PATH = "C:/Users/ilumi/BDA_Project/data/sample_submission.csv"
SAVE_PATH = "C:/Users/ilumi/BDA_Project/data/final_submission.csv"

# test.csv 로딩
test_df = pd.read_csv(TEST_PATH)
test_id = test_df["ID"].copy()  # 나중에 제출용 ID

# 전처리 (train과 동일하게)
test_df.drop(columns=["ID", "generation", "class4"], inplace=True)

# 훈련 데이터 기준 결측률 90% 이상 컬럼 제거
test_df.drop(columns=[col for col in high_null_cols if col in test_df.columns], inplace=True)

# 결측치 처리 및 인코딩
test_df.fillna("무응답", inplace=True)
for col in test_df.columns:
    test_df[col] = test_df[col].astype("category").cat.codes

# 예측
test_preds = rf.predict(test_df)

# ----------------- 14. 제출 파일 저장 ------------------
submission = pd.read_csv(SUBMIT_PATH)
submission["withdrawal"] = test_preds
submission.to_csv(SAVE_PATH, index=False)

print(f"\n📁 제출 파일이 저장되었습니다: {SAVE_PATH}")