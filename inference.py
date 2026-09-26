# # inference.py
# import polars as pl
# import numpy as np
# import lightgbm as lgb
# from sklearn.feature_extraction.text import TfidfVectorizer
# from features import extract_pair_features
# from concurrent.futures import ProcessPoolExecutor
# import os
# import time

# def process_batch(batch_data):
#     s1_ids, cand_s2_ids_list, s1_names, s1_addrs, s2_names, s2_addrs, model_file = batch_data
#     model = lgb.Booster(model_file=model_file)
    
#     batch_results = {}
#     PROB_THRESHOLD = 0.50
#     MAX_S2_MATCHES = 5
    
#     for i in range(len(s1_ids)):
#         s1_id = s1_ids[i]
#         cand_s2_ids = cand_s2_ids_list[i]
        
#         if not cand_s2_ids:
#             continue
            
#         s1_n = s1_names[i]
#         s1_a = s1_addrs[i]
        
#         cand_features = []
#         for j in range(len(cand_s2_ids)):
#             s2_n = s2_names[i][j]
#             s2_a = s2_addrs[i][j]
#             feats = extract_pair_features(s1_n, s1_a, s2_n, s2_a)
#             cand_features.append(feats)
            
#         probs = model.predict(np.array(cand_features))
        
#         matched = []
#         for p, s2_id in sorted(zip(probs, cand_s2_ids), reverse=True):
#             if p >= PROB_THRESHOLD and len(matched) < MAX_S2_MATCHES:
#                 matched.append(s2_id)
        
#         if matched:
#             batch_results[s1_id] = matched
            
#     return batch_results

# if __name__ == "__main__":
#     print("--- GENERATING FINAL TEST PREDICTIONS (FAST MULTI-CORE) ---")

#     model_path = "matcher_s2_model.txt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError("Model file not found! Run train_model.py first.")

#     print("Loading test datasets...")
#     test_s1 = pl.read_csv("dataset/test/test_source1.tsv", separator="\t", ignore_errors=True)
#     test_s2 = pl.read_csv("dataset/test/test_source2.tsv", separator="\t", ignore_errors=True)

#     print(f"Test S1 Count: {len(test_s1):,}, Test S2 Count: {len(test_s2):,}")

#     test_s1 = test_s1.with_columns(
#         clean_name=pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ")
#     )
#     test_s2 = test_s2.with_columns(
#         clean_name=pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ")
#     )

#     s1_rows = {row["entity_id"]: row for row in test_s1.iter_rows(named=True)}
#     s2_rows = {row["entity_id"]: row for row in test_s2.iter_rows(named=True)}

#     countries = test_s1["country"].unique().to_list()
#     results = {}

#     t0 = time.time()
#     print("\nRunning Multi-Threaded Inference Pipeline...")

#     for country in countries:
#         if not country:
#             continue
        
#         sub_s1 = test_s1.filter(pl.col("country") == country)
#         sub_s2 = test_s2.filter(pl.col("country") == country)
        
#         if len(sub_s1) == 0 or len(sub_s2) == 0:
#             continue
            
#         print(f"Processing Country '{country}' (S1: {len(sub_s1):,}, S2: {len(sub_s2):,})...")
        
#         vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1)
#         tfidf_s1 = vectorizer.fit_transform(sub_s1["clean_name"].to_list())
#         tfidf_s2 = vectorizer.transform(sub_s2["clean_name"].to_list())
        
#         sim_matrix = tfidf_s1.dot(tfidf_s2.T)
        
#         s1_ids = sub_s1["entity_id"].to_list()
#         s2_ids = sub_s2["entity_id"].to_list()
        
#         # Prepare batches for multi-core execution
#         BATCH_SIZE = 5000
#         batches = []
        
#         K = 15
#         for b_start in range(0, len(s1_ids), BATCH_SIZE):
#             b_end = min(b_start + BATCH_SIZE, len(s1_ids))
            
#             b_s1_ids = []
#             b_cand_s2_ids = []
#             b_s1_names, b_s1_addrs = [], []
#             b_s2_names, b_s2_addrs = [], []
            
#             for i in range(b_start, b_end):
#                 row = sim_matrix.getrow(i)
#                 if row.nnz > 0:
#                     top_k = row.indices[np.argsort(row.data)[-K:]]
#                     s1_id = s1_ids[i]
#                     s1_rec = s1_rows[s1_id]
                    
#                     b_s1_ids.append(s1_id)
#                     b_s1_names.append(s1_rec.get("business_name") or "")
#                     b_s1_addrs.append(s1_rec.get("business_address") or "")
                    
#                     cand_ids = [s2_ids[idx] for idx in top_k]
#                     cand_names = [s2_rows[sid].get("business_name") or "" for sid in cand_ids]
#                     cand_addrs = [s2_rows[sid].get("business_address") or "" for sid in cand_ids]
                    
#                     b_cand_s2_ids.append(cand_ids)
#                     b_s2_names.append(cand_names)
#                     b_s2_addrs.append(cand_addrs)
            
#             if b_s1_ids:
#                 batches.append((b_s1_ids, b_cand_s2_ids, b_s1_names, b_s1_addrs, b_s2_names, b_s2_addrs, model_path))

#         print(f"--> Parallelizing {len(batches)} batches across all CPU cores...")
#         with ProcessPoolExecutor() as executor:
#             futures = [executor.submit(process_batch, batch) for batch in batches]
#             for future in futures:
#                 res = future.result()
#                 results.update(res)

#     print(f"\nInference completed in {time.time() - t0:.2f} seconds!")

#     print("Formatting submission TSV file...")
#     out_s1 = []
#     out_matches = []

#     for s1_id in test_s1["entity_id"].to_list():
#         out_s1.append(s1_id)
#         matches = results.get(s1_id, [])
#         out_matches.append(",".join(matches) if matches else "")

#     submission_df = pl.DataFrame({
#         "source1_entity_id": out_s1,
#         "matched_entity_ids": out_matches
#     })

#     output_file = "submission.tsv"
#     submission_df.write_csv(output_file, separator="\t")

#     print(f"\nSUCCESS! Submission saved to '{output_file}'.")
#     print(f"Total S1 entities processed: {len(submission_df):,}")
#     print(f"Total S1 entities with matches found: {len([m for m in out_matches if m]):,}")