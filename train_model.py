# train_model.py
import duckdb
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import os
import time

from features import build_21_features

print("=== FAST LIGHTGBM TRAINING (21-FEATURE VECTOR) ===")
t0 = time.time()

con = duckdb.connect()
con.execute("SET memory_limit = '8GB';")

TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"

if not os.path.exists(TRAIN_S1) or not os.path.exists(TRAIN_GROUND_TRUTH):
    print(f"[!] Training files missing in dataset/train/. Exiting.")
    exit(1)

# 1. Load Ground Truth and Source Tables
print("\n1/4 Loading training datasets in DuckDB...")
con.execute(f"""
    CREATE TABLE gt AS
    SELECT source1_entity_id AS s1_id, unnest(string_split(matched_entity_ids, ',')) AS target_id
    FROM read_csv('{TRAIN_GROUND_TRUTH}', delim='\t', header=True, auto_detect=True);
""")

con.execute(f"""
    CREATE TABLE s1 AS
    SELECT entity_id AS s1_id, business_name, business_address, country
    FROM read_csv('{TRAIN_S1}', delim='\t', header=True, auto_detect=True);
""")

con.execute(f"""
    CREATE TABLE s2 AS
    SELECT entity_id AS s2_id, business_name, business_address, country
    FROM read_csv('{TRAIN_S2}', delim='\t', header=True, auto_detect=True);
""")

# 2. Sample 150k Positive & 450k Negative Pairs
print("\n2/4 Sampling 150k positive + 450k hard negative pairs...")
con.execute("""
    CREATE TABLE pos_pairs AS
    SELECT 
        s1.business_name AS name1, s2.business_name AS name2,
        s1.business_address AS addr1, s2.business_address AS addr2,
        s1.country AS c1, s2.country AS c2,
        1 AS label
    FROM gt
    JOIN s1 ON gt.s1_id = s1.s1_id
    JOIN s2 ON gt.target_id = s2.s2_id
    USING SAMPLE 150000;
""")

con.execute("""
    CREATE TABLE neg_pairs AS
    SELECT 
        s1.business_name AS name1, s2.business_name AS name2,
        s1.business_address AS addr1, s2.business_address AS addr2,
        s1.country AS c1, s2.country AS c2,
        0 AS label
    FROM s1
    JOIN s2 ON s1.country = s2.country 
           AND substring(s1.business_name, 1, 3) = substring(s2.business_name, 1, 3)
    LEFT JOIN gt ON s1.s1_id = gt.s1_id AND s2.s2_id = gt.target_id
    WHERE gt.target_id IS NULL
    USING SAMPLE 450000;
""")

all_pairs = con.execute("""
    SELECT name1, name2, addr1, addr2, c1, c2, label FROM pos_pairs
    UNION ALL
    SELECT name1, name2, addr1, addr2, c1, c2, label FROM neg_pairs;
""").fetchall()

print(f"--> Total training sample pairs: {len(all_pairs):,}")

# 3. Extract 21 Features Vectorized
print("\n3/4 Extracting 21-feature vectors...")
X = []
y = []

for row in all_pairs:
    n1, n2, a1, a2, c1, c2, label = row
    feat = build_21_features(n1, n2, a1, a2, c1, c2, source_id="s2")
    X.append(feat)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 4. Train LightGBM
print("\n4/4 Training LightGBM Classifier...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 250,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'random_state': 42,
    'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
)

# Validation Metrics
preds_prob = model.predict_proba(X_val)[:, 1]
preds_bin = (preds_prob >= 0.50).astype(int)

prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds_bin, average='binary')
f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0

print(f"\n================ VALIDATION RESULTS ================")
print(f"--> Precision @ 0.50 : {prec:.4f}")
print(f"--> Recall    @ 0.50 : {rec:.4f}")
print(f"--> F1-Score  @ 0.50 : {f1:.4f}")
print(f"--> F0.5-Score @ 0.50 : {f05:.4f}")
print(f"=====================================================")

# Save model files
model.booster_.save_model("matcher_s2_model.txt")
model.booster_.save_model("matcher_s3_model.txt")
print(f"\nSaved updated 21-feature models to matcher_s2_model.txt & matcher_s3_model.txt in {time.time() - t0:.2f}s.")