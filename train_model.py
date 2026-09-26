# train_model.py
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from features import extract_pair_features
import time

print("--- TRAINING LIGHTGBM MATCHING MODEL ---")

# 1. Load Ground Truth and Source Samples
print("Loading ground truth and training pairs...")
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=15000, ignore_errors=True)

gt_pairs = set()
target_s1 = set()
target_s2 = set()

for row in gt.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    if row["matched_entity_ids"]:
        matches = row["matched_entity_ids"].split(",")
        for match in matches:
            if match.startswith("S2-"):
                gt_pairs.add((s1_id, match))
                target_s1.add(s1_id)
                target_s2.add(match)

s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", ignore_errors=True).filter(
    pl.col("entity_id").is_in(list(target_s1))
)
s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", ignore_errors=True).filter(
    pl.col("entity_id").is_in(list(target_s2))
)

# Convert to dictionaries for fast key-value lookups
s1_dict = {row["entity_id"]: row for row in s1.iter_rows(named=True)}
s2_dict = {row["entity_id"]: row for row in s2.iter_rows(named=True)}

# 2. TF-IDF Blocking to generate candidate dataset
s1_clean = [row.get("business_name") or "" for row in s1.iter_rows(named=True)]
s2_clean = [row.get("business_name") or "" for row in s2.iter_rows(named=True)]

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1)
tfidf_s1 = vectorizer.fit_transform(s1_clean)
tfidf_s2 = vectorizer.transform(s2_clean)

sim_matrix = tfidf_s1.dot(tfidf_s2.T)

s1_ids = list(s1_dict.keys())
s2_ids = list(s2_dict.keys())

X_data = []
y_data = []

print("Extracting similarity features for candidate pairs...")
K = 15
for i in range(len(s1_ids)):
    row = sim_matrix.getrow(i)
    if row.nnz > 0:
        top_k = row.indices[np.argsort(row.data)[-K:]]
        s1_id = s1_ids[i]
        s1_rec = s1_dict[s1_id]
        
        for idx in top_k:
            s2_id = s2_ids[idx]
            s2_rec = s2_dict[s2_id]
            
            # Extract Features
            feats = extract_pair_features(
                s1_rec.get("business_name"), s1_rec.get("business_address"),
                s2_rec.get("business_name"), s2_rec.get("business_address")
            )
            
            # Label: 1 if true positive pair else 0
            label = 1 if (s1_id, s2_id) in gt_pairs else 0
            
            X_data.append(feats)
            y_data.append(label)

X = np.array(X_data)
y = np.array(y_data)

print(f"Feature dataset shape: {X.shape}, Positive labels: {sum(y)}")

# 3. Train LightGBM Model
print("Training LightGBM classifier...")
train_data = lgb.Dataset(X, label=y)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

model = lgb.train(params, train_data, num_boost_round=100)
model.save_model("matcher_s2_model.txt")
print("Model trained and saved to 'matcher_s2_model.txt' successfully!")