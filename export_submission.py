# export_submission.py
import duckdb
import polars as pl
import os
import time

print("--- EXPORT SUBMISSION (WITH GLOBAL MATCH CAPPING) ---")
t0 = time.time()

con = duckdb.connect()
con.execute("SET memory_limit = '10GB';")
con.execute("PRAGMA temp_directory='work/duckdb_tmp';")

S1_PATH = "dataset/test/test_source1.tsv"

# 1. Load S1
con.execute(f"""
    CREATE OR REPLACE TABLE test_s1 AS
    SELECT entity_id
    FROM read_csv('{S1_PATH}', delim='\t', header=True, auto_detect=True);
""")

# 2. Extract S2 (capped at 5)
s2_glob = "work/results_s2/*.parquet"
if os.path.exists("work/results_s2") and len(os.listdir("work/results_s2")) > 0:
    con.execute(f"""
        CREATE OR REPLACE TABLE ranked_s2 AS
        SELECT s1_id, s2_id, prob
        FROM read_parquet('{s2_glob}')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) <= 5;
    """)
else:
    con.execute("CREATE TABLE ranked_s2 (s1_id VARCHAR, s2_id VARCHAR, prob DOUBLE);")

# 3. Extract S3 (capped at 6)
s3_glob = "work/results_s3/*.parquet"
if os.path.exists("work/results_s3") and len(os.listdir("work/results_s3")) > 0:
    con.execute(f"""
        CREATE OR REPLACE TABLE ranked_s3 AS
        SELECT s1_id, s2_id, prob
        FROM read_parquet('{s3_glob}')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) <= 6;
    """)
else:
    con.execute("CREATE TABLE ranked_s3 (s1_id VARCHAR, s2_id VARCHAR, prob DOUBLE);")

# 4. Merge S2 + S3 AND APPLY GLOBAL TOP-6 PROBABILITY CAP
GLOBAL_CAP = 6  # Keeps only the top 6 highest probability matches combined

con.execute(f"""
    CREATE OR REPLACE TABLE all_matches AS
    SELECT s1_id, s2_id, prob
    FROM (
        SELECT s1_id, s2_id, prob FROM ranked_s2
        UNION ALL
        SELECT s1_id, s2_id, prob FROM ranked_s3
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) <= {GLOBAL_CAP};
""")

con.execute("""
    CREATE OR REPLACE TABLE per_s1_agg AS
    SELECT s1_id AS source1_entity_id,
           string_agg(s2_id, ',' ORDER BY prob DESC) AS matched_entity_ids
    FROM all_matches
    GROUP BY s1_id;
""")

# 5. Write submission.tsv
submission = con.execute("""
    SELECT s1.entity_id AS source1_entity_id,
           coalesce(agg.matched_entity_ids, '') AS matched_entity_ids
    FROM test_s1 s1
    LEFT JOIN per_s1_agg agg ON agg.source1_entity_id = s1.entity_id
""").pl()

submission.write_csv("submission.tsv", separator="\t")

n_matched = submission.filter(pl.col("matched_entity_ids") != "").height
print(f"\nSUCCESS! Clean submission.tsv written in {time.time() - t0:.2f}s.")
print(f"Total S1 Entities Processed: {len(submission):,}")
print(f"Total S1 Entities Matched: {n_matched:,}")