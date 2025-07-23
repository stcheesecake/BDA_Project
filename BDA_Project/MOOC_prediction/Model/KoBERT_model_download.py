# download_kobert.py
from transformers import AutoTokenizer, AutoModel

save_dir = "./models/kobert-base-v1"

# KoBERT 모델과 토크나이저 다운로드 후 저장
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
tokenizer.save_pretrained(save_dir)

model = AutoModel.from_pretrained("skt/kobert-base-v1")
model.save_pretrained(save_dir)

print(f"KoBERT 저장 완료: {save_dir}")