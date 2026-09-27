# fast_inference.py
#
# Memory-safe, out-of-core entity resolution inference pipeline using 21-feature vectors.

import duckdb
import polars as pl
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import time
import glob

from features import build_21_features

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DB_MEMORY_LIMIT = "4GB"
DB_THREADS = 4                          # DuckDB SQL execution threads
WORK_DIR = "work"                       # Scratch directory for parquet files
N_WORKERS = 4                           # Parallel worker processes

CHUNK_SIZE = 10_000                     # Rows per chunk file
PROB_THRESHOLD = 0.50                   # Probability cutoff for match candidate

DRY_RUN_S1_LIMIT = None                 # Set to integer (e.g. 50000) for testing

# Multi-pass Sorted Neighborhood Method configuration
SNM_PASSES = {
    "name":         {"window": 25},   # Normalized business name
    "token_sorted": {"window": 15},   # Alphabetized word tokens
    "address":      {"window": 15},   # Normalized business address
}

SOURCES = {
    "s2": {
        "model_path": "matcher_s2_model.txt",
        "test_path": "dataset/test/test_source2.tsv",
        "max_matches": 5,
    },
    "s3": {
        "model_path": "matcher_s3_model.txt",
        "test_path": "dataset/test/test_source3.tsv",
        "max_matches": 6,
    },
}
S1_PATH = "dataset/test/test_source1.tsv"

# Worker global model holder
_WORKER_MODEL = None

def _init_worker(model_path):
    global _WORKER_MODEL
    _WORKER_MODEL = lgb.Booster(model_file=model_path)

def _process_chunk_file(args):
    """Runs inside worker process. Reads Parquet chunk, computes 21 features, scores via LightGBM."""
    chunk_path, source_id = args
    df = pl.read_parquet(chunk_path)

    # Compute 21-feature vectors for every candidate pair in chunk
    feats = [
        build_21_features(
            r["s1_name"], r["s2_name"],
            r["s1_addr"], r["s2_addr"],
            r["s1_country"], r["s2_country"],
            source_id=source_id
        )
        for r in df.iter_rows(named=True)
    ]

    probs = _WORKER_MODEL.predict(np.array(feats, dtype=np.float32))

    out = df.select(["s1_id", "s2_id"]).with_columns(
        pl.Series("prob", probs)
    ).filter(pl.col("prob") >= PROB_THRESHOLD)

    out_path = chunk_path.replace("candidates", "results")
    out.write_parquet(out_path)
    return out_path

# ----------------------------------------------------------------------------
# Blocking: Multi-Pass Sorted Neighborhood Method in DuckDB
# ----------------------------------------------------------------------------
def _sort_key_sql(kind, name_col="business_name", addr_col="business_address"):
    norm_name = f"trim(lower(regexp_replace(coalesce({name_col}, ''), '[^a-zA-Z0-9 ]', '', 'g')))"
    if kind == "name":
        return norm_name
    if kind == "token_sorted":
        return f"array_to_string(list_sort(str_split({norm_name}, ' ')), ' ')"
    if kind == "address":
        return f"trim(lower(regexp_replace(coalesce({addr_col}, ''), '[^a-zA-Z0-9 ]', '', 'g')))"
    raise ValueError(kind)

MAX_CANDIDATES_PER_S1_PER_PASS = 40

def build_candidates(con, s1_table, s2_table, dest_table):
    con.execute(f"CREATE OR REPLACE TABLE {dest_table} (s1_id VARCHAR, s2_id VARCHAR);")

    for kind, cfg in SNM_PASSES.items():
        t_pass = time.time()
        window = cfg["window"]
        sort_expr_s1 = _sort_key_sql(kind)
        sort_expr_s2 = _sort_key_sql(kind)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE pass_ranked AS
            SELECT entity_id, country, src,
                   (ROW_NUMBER() OVER (PARTITION BY country ORDER BY sort_key, entity_id) - 1) // {window} AS bucket
            FROM (
                SELECT entity_id, country, {sort_expr_s1} AS sort_key, 'A' AS src
                FROM {s1_table}
                UNION ALL
                SELECT entity_id, country, {sort_expr_s2} AS sort_key, 'B' AS src
                FROM {s2_table}
            );
        """)

        con.execute(f"""
            INSERT INTO {dest_table}
            SELECT s1_id, s2_id FROM (
                SELECT
                    CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END AS s1_id,
                    CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END AS s2_id
                FROM pass_ranked a
                JOIN pass_ranked b
                  ON a.country = b.country AND a.bucket = b.bucket AND a.src <> b.src

                UNION ALL

                SELECT
                    CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END,
                    CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END
                FROM pass_ranked a
                JOIN pass_ranked b
                  ON a.country = b.country AND a.bucket + 1 = b.bucket AND a.src <> b.src
            ) raw
            QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id) <= {MAX_CANDIDATES_PER_S1_PER_PASS};
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
        print(f"    [{kind}] pass completed in {time.time() - t_pass:.1f}s, cumulative candidates: {n:,}")

    con.execute(f"CREATE OR REPLACE TABLE {dest_table} AS SELECT DISTINCT * FROM {dest_table};")
    n = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
    print(f"    Total deduped candidate pairs: {n:,}")

def write_candidate_chunks(con, cand_table, s1_table, s2_table, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    con.execute(f"""
        CREATE OR REPLACE TABLE {cand_table} AS
        SELECT s1_id, s2_id, (ROW_NUMBER() OVER () - 1) // {CHUNK_SIZE} AS chunk_id
        FROM {cand_table};
    """)
    n_chunks = con.execute(f"SELECT COALESCE(MAX(chunk_id), -1) + 1 FROM {cand_table}").fetchone()[0]
    print(f"    Streaming {n_chunks:,} candidate chunk files to {out_dir}/...")

    files = []
    t_chunks = time.time()
    for cid in range(n_chunks):
        path = os.path.join(out_dir, f"chunk_{cid}.parquet")
        con.execute(f"""
            COPY (
                SELECT c.s1_id, c.s2_id,
                       s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
                       s2.business_name AS s2_name, s2.business_address AS s2_addr, s2.country AS s2_country
                FROM {cand_table} c
                JOIN {s1_table} s1 ON s1.entity_id = c.s1_id
                JOIN {s2_table} s2 ON s2.entity_id = c.s2_id
                WHERE c.chunk_id = {cid}
            ) TO '{path}' (FORMAT PARQUET);
        """)
        files.append(path)
        if (cid + 1) % 100 == 0:
            print(f"      {cid + 1:,}/{n_chunks:,} chunks written ({time.time() - t_chunks:.1f}s elapsed)")

    return files

def run_source(con, source_name, cfg, s1_table):
    print(f"\n=== Running Source Catalog: {source_name.upper()} ===")
    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        fallback = SOURCES["s2"]["model_path"]
        print(f"  [!] {model_path} missing, using {fallback}")
        model_path = fallback

    s_table = f"test_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TABLE {s_table} AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{cfg["test_path"]}', delim='\t', header=True, auto_detect=True);
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {s_table}").fetchone()[0]
    print(f"  Loaded {n:,} {source_name} records")

    cand_table = f"candidates_{source_name}"
    build_candidates(con, s1_table, s_table, cand_table)

    chunk_dir = os.path.join(WORK_DIR, f"candidates_{source_name}")
    chunk_files = write_candidate_chunks(con, cand_table, s1_table, s_table, chunk_dir)

    if not chunk_files:
        return f"results_{source_name}"

    result_dir = os.path.join(WORK_DIR, f"results_{source_name}")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    print(f"  Scoring {len(chunk_files):,} chunks across {N_WORKERS} workers using 21-feature LightGBM...")
    t0 = time.time()
    
    # Pack chunk file path along with source catalog string
    worker_args = [(f, source_name) for f in chunk_files]
    
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(model_path,)) as ex:
        for i, path in enumerate(ex.map(_process_chunk_file, worker_args, chunksize=4)):
            if (i + 1) % 100 == 0:
                print(f"    {i + 1:,}/{len(chunk_files):,} chunks scored ({time.time() - t0:.1f}s elapsed)")

    print(f"  Scoring finished in {time.time() - t0:.1f}s")
    return result_dir

def rank_and_cap(con, result_glob, max_matches, out_table):
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT s1_id, s2_id, prob
        FROM read_parquet('{result_glob}')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) <= {max_matches};
    """)

def main():
    t0 = time.time()
    os.makedirs(WORK_DIR, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DB_MEMORY_LIMIT}';")
    con.execute(f"SET threads = {DB_THREADS};")
    con.execute("SET preserve_insertion_order = false;")

    con.execute(f"""
        CREATE OR REPLACE TABLE test_s1 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{S1_PATH}', delim='\t', header=True, auto_detect=True);
    """)
    n_s1 = con.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]
    print(f"Loaded {n_s1:,} S1 records in {time.time() - t0:.2f}s")

    ranked_tables = {}
    for source_name, cfg in SOURCES.items():
        result_dir = run_source(con, source_name, cfg, "test_s1")
        result_glob = os.path.join(result_dir, "*.parquet")
        if glob.glob(result_glob):
            ranked_table = f"ranked_{source_name}"
            rank_and_cap(con, result_glob, cfg["max_matches"], ranked_table)
            ranked_tables[source_name] = ranked_table
        else:
            ranked_tables[source_name] = None

    print("\nAssembling final submission (S2 + S3 combined, ranked by probability)...")
    union_parts = [f"SELECT s1_id, s2_id, prob FROM {tbl}" for tbl in ranked_tables.values() if tbl]

    con.execute(f"""
        CREATE OR REPLACE TABLE all_matches AS
        {' UNION ALL '.join(union_parts)};
    """)

    con.execute("""
        CREATE OR REPLACE TABLE per_s1_agg AS
        SELECT s1_id AS source1_entity_id,
               string_agg(s2_id, ',' ORDER BY prob DESC) AS matched_entity_ids
        FROM all_matches
        GROUP BY s1_id;
    """)

    submission = con.execute("""
        SELECT s1.entity_id AS source1_entity_id,
               coalesce(agg.matched_entity_ids, '') AS matched_entity_ids
        FROM test_s1 s1
        LEFT JOIN per_s1_agg agg ON agg.source1_entity_id = s1.entity_id
    """).pl()

    submission.write_csv("submission.tsv", separator="\t")

    n_matched = submission.filter(pl.col("matched_entity_ids") != "").height
    print(f"\nSUCCESS in {time.time() - t0:.2f}s!")
    print(f"Total S1 entities: {len(submission):,}")
    print(f"S1 entities with matches: {n_matched:,}")

if __name__ == "__main__":
    main()
 















































# # fast_inference.py
# #
# # Memory-safe, out-of-core entity resolution inference pipeline using 21-feature vectors.

# import duckdb
# import polars as pl
# import numpy as np
# import lightgbm as lgb
# from concurrent.futures import ProcessPoolExecutor
# import os
# import shutil
# import time
# import glob

# from features import build_21_features

# # ----------------------------------------------------------------------------
# # CONFIG
# # ----------------------------------------------------------------------------
# DB_MEMORY_LIMIT = "4GB"
# DB_THREADS = 4                          # DuckDB SQL execution threads
# WORK_DIR = "work"                       # Scratch directory for parquet files
# N_WORKERS = 4                           # Parallel worker processes

# CHUNK_SIZE = 10_000                     # Rows per chunk file
# PROB_THRESHOLD = 0.50                   # Probability cutoff for match candidate

# DRY_RUN_S1_LIMIT = None               # Set to integer (e.g. 50000) for testing

# # Multi-pass Sorted Neighborhood Method configuration
# SNM_PASSES = {
#     "name":         {"window": 25},   # Normalized business name
#     "token_sorted": {"window": 15},   # Alphabetized word tokens
#     "address":      {"window": 15},   # Normalized business address
# }

# SOURCES = {
#     "s2": {
#         "model_path": "matcher_s2_model.txt",
#         "test_path": "dataset/test/test_source2.tsv",
#         "max_matches": 5,
#     },
#     "s3": {
#         "model_path": "matcher_s3_model.txt",
#         "test_path": "dataset/test/test_source3.tsv",
#         "max_matches": 6,
#     },
# }
# S1_PATH = "dataset/test/test_source1.tsv"

# # Worker global model holder
# _WORKER_MODEL = None

# def _init_worker(model_path):
#     global _WORKER_MODEL
#     _WORKER_MODEL = lgb.Booster(model_file=model_path)

# def _process_chunk_file(args):
#     """Runs inside worker process. Reads Parquet chunk, computes 21 features, scores via LightGBM."""
#     chunk_path, source_id = args
#     df = pl.read_parquet(chunk_path)

#     # Compute 21-feature vectors for every candidate pair in chunk
#     feats = [
#         build_21_features(
#             r["s1_name"], r["s2_name"],
#             r["s1_addr"], r["s2_addr"],
#             r["s1_country"], r["s2_country"],
#             source_id=source_id
#         )
#         for r in df.iter_rows(named=True)
#     ]

#     probs = _WORKER_MODEL.predict(np.array(feats, dtype=np.float32))

#     out = df.select(["s1_id", "s2_id"]).with_columns(
#         pl.Series("prob", probs)
#     ).filter(pl.col("prob") >= PROB_THRESHOLD)

#     out_path = chunk_path.replace("candidates", "results")
#     out.write_parquet(out_path)
#     return out_path

# # ----------------------------------------------------------------------------
# # Blocking: Multi-Pass Sorted Neighborhood Method in DuckDB
# # ----------------------------------------------------------------------------
# def _sort_key_sql(kind, name_col="business_name", addr_col="business_address"):
#     norm_name = f"trim(lower(regexp_replace(coalesce({name_col}, ''), '[^a-zA-Z0-9 ]', '', 'g')))"
#     if kind == "name":
#         return norm_name
#     if kind == "token_sorted":
#         return f"array_to_string(list_sort(str_split({norm_name}, ' ')), ' ')"
#     if kind == "address":
#         return f"trim(lower(regexp_replace(coalesce({addr_col}, ''), '[^a-zA-Z0-9 ]', '', 'g')))"
#     raise ValueError(kind)

# MAX_CANDIDATES_PER_S1_PER_PASS = 40

# def build_candidates(con, s1_table, s2_table, dest_table):
#     con.execute(f"CREATE OR REPLACE TABLE {dest_table} (s1_id VARCHAR, s2_id VARCHAR);")

#     for kind, cfg in SNM_PASSES.items():
#         t_pass = time.time()
#         window = cfg["window"]
#         sort_expr_s1 = _sort_key_sql(kind)
#         sort_expr_s2 = _sort_key_sql(kind)

#         con.execute(f"""
#             CREATE OR REPLACE TEMP TABLE pass_ranked AS
#             SELECT entity_id, country, src,
#                    (ROW_NUMBER() OVER (PARTITION BY country ORDER BY sort_key, entity_id) - 1) // {window} AS bucket
#             FROM (
#                 SELECT entity_id, country, {sort_expr_s1} AS sort_key, 'A' AS src
#                 FROM {s1_table}
#                 UNION ALL
#                 SELECT entity_id, country, {sort_expr_s2} AS sort_key, 'B' AS src
#                 FROM {s2_table}
#             );
#         """)

#         con.execute(f"""
#             INSERT INTO {dest_table}
#             SELECT s1_id, s2_id FROM (
#                 SELECT
#                     CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END AS s1_id,
#                     CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END AS s2_id
#                 FROM pass_ranked a
#                 JOIN pass_ranked b
#                   ON a.country = b.country AND a.bucket = b.bucket AND a.src <> b.src

#                 UNION ALL

#                 SELECT
#                     CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END,
#                     CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END
#                 FROM pass_ranked a
#                 JOIN pass_ranked b
#                   ON a.country = b.country AND a.bucket + 1 = b.bucket AND a.src <> b.src
#             ) raw
#             QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id) <= {MAX_CANDIDATES_PER_S1_PER_PASS};
#         """)
#         n = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
#         print(f"    [{kind}] pass completed in {time.time() - t_pass:.1f}s, cumulative candidates: {n:,}")

#     con.execute(f"CREATE OR REPLACE TABLE {dest_table} AS SELECT DISTINCT * FROM {dest_table};")
#     n = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
#     print(f"    Total deduped candidate pairs: {n:,}")

# def write_candidate_chunks(con, cand_table, s1_table, s2_table, out_dir):
#     if os.path.exists(out_dir):
#         shutil.rmtree(out_dir)
#     os.makedirs(out_dir, exist_ok=True)

#     con.execute(f"""
#         CREATE OR REPLACE TABLE {cand_table} AS
#         SELECT s1_id, s2_id, (ROW_NUMBER() OVER () - 1) // {CHUNK_SIZE} AS chunk_id
#         FROM {cand_table};
#     """)
#     n_chunks = con.execute(f"SELECT COALESCE(MAX(chunk_id), -1) + 1 FROM {cand_table}").fetchone()[0]
#     print(f"    Streaming {n_chunks:,} candidate chunk files to {out_dir}/...")

#     files = []
#     t_chunks = time.time()
#     for cid in range(n_chunks):
#         path = os.path.join(out_dir, f"chunk_{cid}.parquet")
#         con.execute(f"""
#             COPY (
#                 SELECT c.s1_id, c.s2_id,
#                        s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
#                        s2.business_name AS s2_name, s2.business_address AS s2_addr, s2.country AS s2_country
#                 FROM {cand_table} c
#                 JOIN {s1_table} s1 ON s1.entity_id = c.s1_id
#                 JOIN {s2_table} s2 ON s2.entity_id = c.s2_id
#                 WHERE c.chunk_id = {cid}
#             ) TO '{path}' (FORMAT PARQUET);
#         """)
#         files.append(path)
#         if (cid + 1) % 100 == 0:
#             print(f"      {cid + 1:,}/{n_chunks:,} chunks written ({time.time() - t_chunks:.1f}s elapsed)")

#     return files

# def run_source(con, source_name, cfg, s1_table):
#     print(f"\n=== Running Source Catalog: {source_name.upper()} ===")
#     model_path = cfg["model_path"]
#     if not os.path.exists(model_path):
#         fallback = SOURCES["s2"]["model_path"]
#         print(f"  [!] {model_path} missing, using {fallback}")
#         model_path = fallback

#     s_table = f"test_{source_name}"
#     con.execute(f"""
#         CREATE OR REPLACE TABLE {s_table} AS
#         SELECT entity_id, business_name, business_address, country
#         FROM read_csv('{cfg["test_path"]}', delim='\t', header=True, auto_detect=True);
#     """)
#     n = con.execute(f"SELECT COUNT(*) FROM {s_table}").fetchone()[0]
#     print(f"  Loaded {n:,} {source_name} records")

#     cand_table = f"candidates_{source_name}"
#     build_candidates(con, s1_table, s_table, cand_table)

#     chunk_dir = os.path.join(WORK_DIR, f"candidates_{source_name}")
#     chunk_files = write_candidate_chunks(con, cand_table, s1_table, s_table, chunk_dir)

#     if not chunk_files:
#         return f"results_{source_name}"

#     result_dir = os.path.join(WORK_DIR, f"results_{source_name}")
#     if os.path.exists(result_dir):
#         shutil.rmtree(result_dir)
#     os.makedirs(result_dir, exist_ok=True)

#     print(f"  Scoring {len(chunk_files):,} chunks across {N_WORKERS} workers using 21-feature LightGBM...")
#     t0 = time.time()
    
#     # Pack chunk file path along with source catalog string
#     worker_args = [(f, source_name) for f in chunk_files]
    
#     with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(model_path,)) as ex:
#         for i, path in enumerate(ex.map(_process_chunk_file, worker_args, chunksize=4)):
#             if (i + 1) % 100 == 0:
#                 print(f"    {i + 1:,}/{len(chunk_files):,} chunks scored ({time.time() - t0:.1f}s elapsed)")

#     print(f"  Scoring finished in {time.time() - t0:.1f}s")
#     return result_dir

# # def rank_and_cap(con, result_glob, max_matches, out_table):
# #     con.execute(f"""
# #         CREATE OR REPLACE TABLE {out_table} AS
# #         SELECT s1_id, s2_id, prob
# #         FROM read_parquet('{result_glob}')
# #         QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) <= {max_matches};
# #     """)


# def rank_and_cap(con, result_glob, max_matches, out_table):
#     """Top-K by probability per s1_id using disk-backed sorting."""
#     con.execute(f"""
#         CREATE OR REPLACE TABLE {out_table} AS
#         SELECT s1_id, s2_id, prob
#         FROM read_parquet('{result_glob}')
#         QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY prob DESC) <= {max_matches};
#     """)

# # def main():
# #     t0 = time.time()
# #     os.makedirs(WORK_DIR, exist_ok=True)

# #     con = duckdb.connect()
# #     con.execute(f"SET memory_limit = '{DB_MEMORY_LIMIT}';")
# #     con.execute(f"SET threads = {DB_THREADS};")
# #     con.execute("SET preserve_insertion_order = false;")

# #     con.execute(f"""
# #         CREATE OR REPLACE TABLE test_s1 AS
# #         SELECT entity_id, business_name, business_address, country
# #         FROM read_csv('{S1_PATH}', delim='\t', header=True, auto_detect=True);
# #     """)
# #     n_s1 = con.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]
# #     print(f"Loaded {n_s1:,} S1 records in {time.time() - t0:.2f}s")

# #     ranked_tables = {}
# #     for source_name, cfg in SOURCES.items():
# #         result_dir = run_source(con, source_name, cfg, "test_s1")
# #         result_glob = os.path.join(result_dir, "*.parquet")
# #         if glob.glob(result_glob):
# #             ranked_table = f"ranked_{source_name}"
# #             rank_and_cap(con, result_glob, cfg["max_matches"], ranked_table)
# #             ranked_tables[source_name] = ranked_table
# #         else:
# #             ranked_tables[source_name] = None

# #     print("\nAssembling final submission (S2 + S3 combined, ranked by probability)...")
# #     union_parts = [f"SELECT s1_id, s2_id, prob FROM {tbl}" for tbl in ranked_tables.values() if tbl]

# #     con.execute(f"""
# #         CREATE OR REPLACE TABLE all_matches AS
# #         {' UNION ALL '.join(union_parts)};
# #     """)

# #     con.execute("""
# #         CREATE OR REPLACE TABLE per_s1_agg AS
# #         SELECT s1_id AS source1_entity_id,
# #                string_agg(s2_id, ',' ORDER BY prob DESC) AS matched_entity_ids
# #         FROM all_matches
# #         GROUP BY s1_id;
# #     """)

# #     submission = con.execute("""
# #         SELECT s1.entity_id AS source1_entity_id,
# #                coalesce(agg.matched_entity_ids, '') AS matched_entity_ids
# #         FROM test_s1 s1
# #         LEFT JOIN per_s1_agg agg ON agg.source1_entity_id = s1.entity_id
# #     """).pl()

# #     submission.write_csv("submission.tsv", separator="\t")

# #     n_matched = submission.filter(pl.col("matched_entity_ids") != "").height
# #     print(f"\nSUCCESS in {time.time() - t0:.2f}s!")
# #     print(f"Total S1 entities: {len(submission):,}")
# #     print(f"S1 entities with matches: {n_matched:,}")

# # if __name__ == "__main__":
# #     main()



# def main():
#     t0 = time.time()
#     os.makedirs(WORK_DIR, exist_ok=True)

#     con = duckdb.connect()
#     con.execute(f"SET memory_limit = '{DB_MEMORY_LIMIT}';")
#     con.execute(f"SET threads = {DB_THREADS};")
#     con.execute("SET preserve_insertion_order = false;")

#     # FIX: Enable disk spillover so DuckDB handles large aggregations out-of-core
#     temp_dir = os.path.join(WORK_DIR, "duckdb_tmp")
#     os.makedirs(temp_dir, exist_ok=True)
#     con.execute(f"PRAGMA temp_directory='{temp_dir.replace('\\', '/')}';")

#     limit_clause = f"LIMIT {DRY_RUN_S1_LIMIT}" if DRY_RUN_S1_LIMIT else ""
#     if DRY_RUN_S1_LIMIT:
#         print(f"*** DRY RUN: capping S1 to {DRY_RUN_S1_LIMIT:,} rows. ***")

#     # Load test_s1 into DuckDB
#     con.execute(f"""
#         CREATE OR REPLACE TABLE test_s1 AS
#         SELECT entity_id, business_name, business_address, country
#         FROM read_csv('{S1_PATH}', delim='\t', header=True, auto_detect=True)
#         {limit_clause};
#     """)

#     n_s1 = con.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]
#     print(f"Loaded {n_s1:,} S1 records in {time.time() - t0:.2f}s")

#     ranked_tables = {}
#     for source_name, cfg in SOURCES.items():
#         result_dir = run_source(con, source_name, cfg, "test_s1")
#         result_glob = os.path.join(result_dir, "*.parquet")
#         if glob.glob(result_glob):
#             ranked_table = f"ranked_{source_name}"
#             rank_and_cap(con, result_glob, cfg["max_matches"], ranked_table)
#             ranked_tables[source_name] = ranked_table
#         else:
#             ranked_tables[source_name] = None

#     print("\nAssembling final submission (S2 + S3 combined, ranked by probability)...")
#     union_parts = [f"SELECT s1_id, s2_id, prob FROM {tbl}" for tbl in ranked_tables.values() if tbl]

#     con.execute(f"""
#         CREATE OR REPLACE TABLE all_matches AS
#         {' UNION ALL '.join(union_parts)};
#     """)

#     con.execute("""
#         CREATE OR REPLACE TABLE per_s1_agg AS
#         SELECT s1_id AS source1_entity_id,
#                string_agg(s2_id, ',' ORDER BY prob DESC) AS matched_entity_ids
#         FROM all_matches
#         GROUP BY s1_id;
#     """)

#     submission = con.execute("""
#         SELECT s1.entity_id AS source1_entity_id,
#                coalesce(agg.matched_entity_ids, '') AS matched_entity_ids
#         FROM test_s1 s1
#         LEFT JOIN per_s1_agg agg ON agg.source1_entity_id = s1.entity_id
#     """).pl()

#     submission.write_csv("submission.tsv", separator="\t")

#     n_matched = submission.filter(pl.col("matched_entity_ids") != "").height
#     print(f"\nSUCCESS in {time.time() - t0:.2f}s!")
#     print(f"Total S1 entities: {len(submission):,}")
#     print(f"S1 entities with matches: {n_matched:,}")

# if __name__ == "__main__":
#     main()