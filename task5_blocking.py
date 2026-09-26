# # task5_blocking.py
# import polars as pl
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import csr_matrix
# import time

# print("--- TASK 5: CANDIDATE BLOCKING BENCHMARK ---")

# # 1. Load Data Sample (50,000 rows for high-speed empirical evaluation)
# print("Loading dataset sample...")
# s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=50000, ignore_errors=True)
# s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=50000, ignore_errors=True)
# gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=50000, ignore_errors=True)

# # 2. Extract Valid Pairs from Ground Truth
# gt_pairs = set()
# for row in gt.iter_rows(named=True):
#     s1_id = row["source1_entity_id"]
#     if row["matched_entity_ids"]:
#         matches = row["matched_entity_ids"].split(",")
#         for match in matches:
#             if match.startswith("S2-"):
#                 gt_pairs.add((s1_id, match))

# print(f"Total True Positive S1-S2 Pairs in Sample: {len(gt_pairs)}")

# # 3. Clean Text Vectorization
# s1 = s1.with_columns(
#     clean_name=pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ")
# )
# s2 = s2.with_columns(
#     clean_name=pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ")
# )

# # 4. Method 1: TF-IDF Cosine Blocking (Frequency-Filtered)
# print("\n[Testing Strategy 1] TF-IDF Character N-Gram Similarity...")
# t0 = time.time()

# # Character 3-gram vectorizer with frequency bounds (filters common words like 'inc', 'llc', 'store')
# vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, max_df=0.8)

# s1_names = s1["clean_name"].to_list()
# s2_names = s2["clean_name"].to_list()

# tfidf_s1 = vectorizer.fit_transform(s1_names)
# tfidf_s2 = vectorizer.transform(s2_names)

# # Compute Top-K Nearest Neighbors via Sparse Dot Product
# K = 15  # Keep top 15 candidates per S1
# print(f"Retrieving Top-{K} candidates per S1 entity...")

# # Batch matrix multiplication
# candidates_found = set()
# s1_ids = s1["entity_id"].to_list()
# s2_ids = s2["entity_id"].to_list()

# # Matrix product for similarity scores
# sim_matrix = tfidf_s1.dot(tfidf_s2.T)

# for i in range(len(s1_ids)):
#     row = sim_matrix.getrow(i)
#     if row.nnz > 0:
#         top_k_indices = row.indices[np.argsort(row.data)[-K:]]
#         s1_id = s1_ids[i]
#         for idx in top_k_indices:
#             candidates_found.add((s1_id, s2_ids[idx]))

# # Calculate Metrics
# hits = len(candidates_found.intersection(gt_pairs))
# recall = (hits / len(gt_pairs)) * 100 if gt_pairs else 0
# total_candidates = len(candidates_found)
# avg_cand_per_s1 = total_candidates / len(s1_ids)

# print(f"--> Execution Time: {time.time() - t0:.2f} seconds")
# print(f"--> Total Candidate Pairs Generated: {total_candidates:,}")
# print(f"--> Avg Candidates per S1: {avg_cand_per_s1:.2f}")
# print(f"--> Candidate Recall: {recall:.2f}% ({hits}/{len(gt_pairs)} true pairs captured)")

# print("\nTask 5 benchmark run completed.")






















# task5_blocking.py
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time

print("--- TASK 5: VALIDATING COUNTRY + TF-IDF BLOCKING ---")

# 1. Load Ground Truth and sample valid S1 IDs
print("Loading ground truth and sample data...")
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=20000, ignore_errors=True)

# Extract ground truth mapping for S1 -> S2
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

print(f"Targeting {len(target_s1)} S1 entities and {len(gt_pairs)} true positive S1-S2 pairs...")

# 2. Load matching S1 and S2 records
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", ignore_errors=True).filter(
    pl.col("entity_id").is_in(list(target_s1))
)

s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", ignore_errors=True).filter(
    pl.col("entity_id").is_in(list(target_s2))
)

# Clean text
s1 = s1.with_columns(
    clean_name=pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ")
)
s2 = s2.with_columns(
    clean_name=pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ")
)

# Group by Country partition
s1_countries = s1["country"].unique().to_list()
total_candidates = set()

t0 = time.time()
print("\nRunning TF-IDF Blocking partitioned by Country...")

for country in s1_countries:
    if not country:
        continue
    
    s1_sub = s1.filter(pl.col("country") == country)
    s2_sub = s2.filter(pl.col("country") == country)
    
    if len(s1_sub) == 0 or len(s2_sub) == 0:
        continue

    # Character 3-gram TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1)
    
    tfidf_s1 = vectorizer.fit_transform(s1_sub["clean_name"].to_list())
    tfidf_s2 = vectorizer.transform(s2_sub["clean_name"].to_list())
    
    # Compute similarity matrix per country
    sim_matrix = tfidf_s1.dot(tfidf_s2.T)
    
    s1_ids = s1_sub["entity_id"].to_list()
    s2_ids = s2_sub["entity_id"].to_list()
    
    K = 15  # Keep top 15 candidates per S1
    for i in range(len(s1_ids)):
        row = sim_matrix.getrow(i)
        if row.nnz > 0:
            top_k = row.indices[np.argsort(row.data)[-K:]]
            s1_id = s1_ids[i]
            for idx in top_k:
                total_candidates.add((s1_id, s2_ids[idx]))

# Evaluate Recall
hits = len(total_candidates.intersection(gt_pairs))
recall = (hits / len(gt_pairs)) * 100 if gt_pairs else 0

print(f"\n--> Execution Time: {time.time() - t0:.2f} seconds")
print(f"--> Total Candidate Pairs Generated: {len(total_candidates):,}")
print(f"--> Avg Candidates per S1: {len(total_candidates)/len(s1):.2f}")
print(f"--> Candidate Recall: {recall:.2f}% ({hits}/{len(gt_pairs)} true pairs captured)")