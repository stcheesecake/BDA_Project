import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

TEST_SIZE = 0.2

# ─────────────────────────────
# 1. 데이터 불러오기
# ─────────────────────────────
df = pd.read_csv("C:/Users/ilumi/BDA_Project/data/train.csv")

# ─────────────────────────────
# 2. 제거할 컬럼 제거
# ─────────────────────────────
drop_cols = ["ID", "generation", "class4"]
if "withdrawal" in df.columns:
    y = df["withdrawal"]
    df.drop(columns=drop_cols + ["withdrawal"], inplace=True)
else:
    y = df.iloc[:, -1]  # 마지막 열이 타겟인 경우
    df.drop(columns=drop_cols, inplace=True)

# ─────────────────────────────
# 3. 90% 이상 결측치인 컬럼 제거
# ─────────────────────────────
threshold = 0.9
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio >= threshold].index
df.drop(columns=cols_to_drop, inplace=True)

# ─────────────────────────────
# 4. 전부 object니까 → 결측치는 "무응답"으로 보간
# ─────────────────────────────
df.fillna("무응답", inplace=True)

# ─────────────────────────────
# 5. 인코딩 (object → category → code)
# ─────────────────────────────
for col in df.columns:
    df[col] = df[col].astype("category").cat.codes

# 타겟 인코딩
y = y.astype("category").cat.codes

# ─────────────────────────────
# 6. 학습/검증 분할
# ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=TEST_SIZE, random_state=42)

# ─────────────────────────────
# 7. Decision Tree 학습
# ─────────────────────────────
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# ─────────────────────────────
# 8. 평가 결과 출력
# ─────────────────────────────
y_pred = clf.predict(X_test)
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────
# 9. Feature Importance 시각화
# ─────────────────────────────
importances = clf.feature_importances_
feat_names = df.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feat_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()