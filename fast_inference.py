# scalable_inference.py
#
# Memory-safe, skew-proof entity resolution inference pipeline.
#
# WHAT CHANGED vs. fast_inference.py, and why:
#
#   1. BLOCKING (fixes the RAM explosion / disk thrashing)
#      Old: hash-equality join on `first_word_token`. Any token that many
#      records share (even after stopword filtering: "american", "national",
#      truncated names, etc.) produces count(token)^2 pairs -> cross-product
#      blowup, which is exactly what pushed DuckDB/Windows into pagefile
#      swapping.
#      New: multi-pass Sorted Neighborhood Method (SNM). Sort S1 union S2
#      (within country) by a normalization key, number the rows, and only
#      join rows whose row-number differs by <= WINDOW. Candidate volume is
#      now bounded by (|S1|+|S2|) * WINDOW *regardless of key skew* -- a
#      record that shares its first word with 50,000 others still only gets
#      WINDOW neighbors, not 50,000. Multiple sort keys (raw name, word-order
#      -independent "token-sorted" name, normalized address) are unioned to
#      recover recall a single sort order would miss.
#
#   2. INFERENCE (fixes the IPC / serialization bottleneck)
#      Old: ProcessPoolExecutor.map() ships raw Python strings across the
#      process boundary on every single batch -> pickling cost scales with
#      total candidate volume and dominates wall clock at this scale.
#      New: candidates are streamed to small Parquet chunk files on disk.
#      Each worker gets only a *file path* (a few bytes), reads its chunk
#      independently, computes features + runs the model, and writes its
#      own small result file. RAM per worker is bounded by chunk size, not
#      total candidate count, and there is almost no cross-process data
#      transfer.
#
#   3. CORRECTNESS FIX
#      Old: `if len(results[s1_id]) < MAX_S2_MATCHES: results[s1_id].append(...)`
#      keeps the first N matches *seen*, not the N highest-probability
#      matches. Fixed below with an explicit rank-by-probability step.
#
#   4. S3
#      Same code path, parametrized by source table/model/window/cap, then
#      merged per S1 entity into a single comma-separated column.
#
# Tune the CONFIG block for your hardware. Defaults target ~6-8GB RAM.
 
import duckdb
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import time
import glob
 
# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DB_MEMORY_LIMIT = "4GB"
DB_THREADS = 4                          # Keep DuckDB SQL execution on 4 threads
WORK_DIR = "work"                       # Scratch directory for chunked parquet files

# CRITICAL FIX: Explicitly cap worker processes to 4 so 27 processes don't overload RAM
N_WORKERS = 4                           

CHUNK_SIZE = 10_000                     # Reduced chunk size for low RAM footprint per process
PROB_THRESHOLD = 0.50

# DRY_RUN: cap S1 to 50,000 rows for testing. Set to None for the full dataset.
DRY_RUN_S1_LIMIT = None

# Sorted-neighborhood window per sort-key pass
SNM_PASSES = {
    "name":         {"window": 25},   # Normalized business name
    "token_sorted": {"window": 15},   # Words alphabetized -> reordering-proof
    "address":      {"window": 15},   # Normalized address
}

SOURCES = {
    "s2": {
        "model_path": "matcher_s2_model.txt",
        "test_path": "dataset/test/test_source2.tsv",
        "max_matches": 5,
    },
    "s3": {
        "model_path": "matcher_s3_model.txt",  # Falls back to s2 model if absent
        "test_path": "dataset/test/test_source3.tsv",
        "max_matches": 6,
    },
}
S1_PATH = "dataset/test/test_source1.tsv"
 
 
# ----------------------------------------------------------------------------
# Worker-side globals (populated once per process by the pool initializer,
# never re-sent per task -- this is the fix for the IPC bottleneck)
# ----------------------------------------------------------------------------
_WORKER_MODEL = None
 
 
def _init_worker(model_path):
    global _WORKER_MODEL
    _WORKER_MODEL = lgb.Booster(model_file=model_path)
 
 
def _pair_features(s1_n, s1_a, s2_n, s2_a):
    s1_n, s1_a = s1_n or "", s1_a or ""
    s2_n, s2_a = s2_n or "", s2_a or ""
 
    n_ratio = fuzz.ratio(s1_n, s2_n)
    n_partial = fuzz.partial_ratio(s1_n, s2_n)
    n_sort = fuzz.token_sort_ratio(s1_n, s2_n)
    n_set = fuzz.token_set_ratio(s1_n, s2_n)
 
    a_ratio = fuzz.ratio(s1_a, s2_a)
    a_set = fuzz.token_set_ratio(s1_a, s2_a)
 
    import re
    s1_nums = set(re.findall(r"\d+", s1_a))
    s2_nums = set(re.findall(r"\d+", s2_a))
    num_match = 1.0 if s1_nums and s2_nums and (s1_nums & s2_nums) else 0.0
 
    len_diff_n = abs(len(s1_n) - len(s2_n))
    len_diff_a = abs(len(s1_a) - len(s2_a))
    missing_addr = 1.0 if not s1_a or not s2_a else 0.0
 
    return [n_ratio, n_partial, n_sort, n_set, a_ratio, a_set,
            num_match, len_diff_n, len_diff_a, missing_addr]
 
 
def _process_chunk_file(chunk_path):
    """Runs in a worker process. Reads its own chunk from disk, scores it,
    writes its own result file. Only the file path crosses the IPC boundary."""
    df = pl.read_parquet(chunk_path)
 
    feats = [
        _pair_features(r["s1_name"], r["s1_addr"], r["s2_name"], r["s2_addr"])
        for r in df.iter_rows(named=True)
    ]
    probs = _WORKER_MODEL.predict(np.array(feats, dtype=np.float64))
 
    out = df.select(["s1_id", "s2_id"]).with_columns(
        pl.Series("prob", probs)
    ).filter(pl.col("prob") >= PROB_THRESHOLD)
 
    out_path = chunk_path.replace("candidates", "results")
    out.write_parquet(out_path)
    return out_path
 
 
# ----------------------------------------------------------------------------
# Blocking: multi-pass Sorted Neighborhood Method in DuckDB
# ----------------------------------------------------------------------------
def _sort_key_sql(kind, name_col="business_name", addr_col="business_address"):
    norm_name = (
        f"trim(lower(regexp_replace(coalesce({name_col}, ''), "
        f"'[^a-zA-Z0-9 ]', '', 'g')))"
    )
    if kind == "name":
        return norm_name
    if kind == "token_sorted":
        return f"array_to_string(list_sort(str_split({norm_name}, ' ')), ' ')"
    if kind == "address":
        return (
            f"trim(lower(regexp_replace(coalesce({addr_col}, ''), "
            f"'[^a-zA-Z0-9 ]', '', 'g')))"
        )
    raise ValueError(kind)
 
 
# Safety net: no matter how skewed the data is, never keep more than this
# many candidates per S1 entity per pass. Applied via a window function
# *after* the (cheap, hash-joinable) bucket join, so it costs almost nothing
# but hard-caps worst-case output size.
MAX_CANDIDATES_PER_S1_PER_PASS = 40
 
 
def build_candidates(con, s1_table, s2_table, dest_table):
    """Populates `dest_table(s1_id, s2_id)` with deduped candidate pairs.
 
    Uses BUCKETED equi-joins, not the earlier UNNEST(generate_series(...))
    cross-join (which materializes N*W rows before joining -- slow) and not
    an OR'd multi-key join (which DuckDB cannot hash-join at all, and falls
    back to a nested-loop scan over the full cross product -- this is what
    was pinning your CPU/RAM with no progress).
 
    Method: sort S1+S2 together per country, number the rows, then group
    rows into fixed-size buckets of `window` consecutive rows. Two rows
    within `window` of each other in sort order are ALWAYS either in the
    same bucket or in adjacent buckets (provable: for |rn_a - rn_b| <= W,
    floor(rn_a/W) and floor(rn_b/W) differ by at most 1). So "same bucket"
    UNION ALL "adjacent bucket" is an exact superset of the windowed pairs,
    and both are single-condition equality joins -> real hash joins -> fast.
    """
    con.execute(f"CREATE OR REPLACE TABLE {dest_table} (s1_id VARCHAR, s2_id VARCHAR);")
 
    for kind, cfg in SNM_PASSES.items():
        t_pass = time.time()
        window = cfg["window"]
        sort_expr_s1 = _sort_key_sql(kind)
        sort_expr_s2 = _sort_key_sql(kind)
 
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE pass_ranked AS
            SELECT entity_id, country, src,
                   (ROW_NUMBER() OVER (PARTITION BY country ORDER BY sort_key, entity_id) - 1)
                       // {window} AS bucket
            FROM (
                SELECT entity_id, country, {sort_expr_s1} AS sort_key, 'A' AS src
                FROM {s1_table}
                UNION ALL
                SELECT entity_id, country, {sort_expr_s2} AS sort_key, 'B' AS src
                FROM {s2_table}
            );
        """)
        n_ranked = con.execute("SELECT COUNT(*) FROM pass_ranked").fetchone()[0]
        print(f"    [{kind}] ranked {n_ranked:,} rows in {time.time() - t_pass:.1f}s")
 
        con.execute(f"""
            INSERT INTO {dest_table}
            SELECT s1_id, s2_id FROM (
                SELECT
                    CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END AS s1_id,
                    CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END AS s2_id
                FROM pass_ranked a
                JOIN pass_ranked b
                  ON a.country = b.country
                 AND a.bucket = b.bucket
                 AND a.src <> b.src
 
                UNION ALL
 
                SELECT
                    CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END,
                    CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END
                FROM pass_ranked a
                JOIN pass_ranked b
                  ON a.country = b.country
                 AND a.bucket + 1 = b.bucket
                 AND a.src <> b.src
            ) raw
            QUALIFY ROW_NUMBER() OVER (PARTITION BY s1_id) <= {MAX_CANDIDATES_PER_S1_PER_PASS};
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
        print(f"    [{kind}] pass done in {time.time() - t_pass:.1f}s, "
              f"bucket={window}, cumulative candidates={n:,}")
 
    con.execute(f"CREATE OR REPLACE TABLE {dest_table} AS SELECT DISTINCT * FROM {dest_table};")
    n = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
    print(f"    total deduped candidates: {n:,}")
 
 
def write_candidate_chunks(con, cand_table, s1_table, s2_table, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
 
    # Assign chunk ids on the *lightweight* id-only candidate table (just two
    # VARCHAR columns) -- not on the text-joined table. This keeps the row
    # numbering pass cheap regardless of how large the full run's candidate
    # set gets.
    con.execute(f"""
        CREATE OR REPLACE TABLE {cand_table} AS
        SELECT s1_id, s2_id, (ROW_NUMBER() OVER () - 1) // {CHUNK_SIZE} AS chunk_id
        FROM {cand_table};
    """)
    n_chunks = con.execute(
        f"SELECT COALESCE(MAX(chunk_id), -1) + 1 FROM {cand_table}"
    ).fetchone()[0]
    print(f"    streaming {n_chunks:,} chunk files (chunk_size={CHUNK_SIZE:,}) "
          f"to {out_dir}/ ...")
 
    # Write ONE chunk at a time with a plain COPY. No Hive-partitioned writer
    # (that's what kept every partition's buffer open in memory at once and
    # caused the OOM), no full materialization of the text-joined data --
    # each iteration only ever touches CHUNK_SIZE rows of text.
    files = []
    t_chunks = time.time()
    for cid in range(n_chunks):
        path = os.path.join(out_dir, f"chunk_{cid}.parquet")
        con.execute(f"""
            COPY (
                SELECT c.s1_id, c.s2_id,
                       s1.business_name AS s1_name, s1.business_address AS s1_addr,
                       s2.business_name AS s2_name, s2.business_address AS s2_addr
                FROM {cand_table} c
                JOIN {s1_table} s1 ON s1.entity_id = c.s1_id
                JOIN {s2_table} s2 ON s2.entity_id = c.s2_id
                WHERE c.chunk_id = {cid}
            ) TO '{path}' (FORMAT PARQUET);
        """)
        files.append(path)
        if (cid + 1) % 100 == 0:
            print(f"      {cid + 1:,}/{n_chunks:,} chunk files written "
                  f"({time.time() - t_chunks:.1f}s elapsed)")
 
    print(f"    wrote {len(files):,} candidate chunk files to {out_dir}/ "
          f"in {time.time() - t_chunks:.1f}s")
    return files
 
 
# # ----------------------------------------------------------------------------
# # Per-source pipeline
# # ----------------------------------------------------------------------------
# def run_source(con, source_name, cfg, s1_table):
#     print(f"\n=== Source: {source_name} ===")
#     model_path = cfg["model_path"]
#     if not os.path.exists(model_path):
#         fallback = SOURCES["s2"]["model_path"]
#         print(f"  [!] {model_path} not found, falling back to {fallback} "
#               f"(retrain a dedicated model for {source_name} if time allows -- "
#               f"it has different noise characteristics per your notes)")
#         model_path = fallback
 
#     s_table = f"test_{source_name}"
#     con.execute(f"""
#         CREATE OR REPLACE TABLE {s_table} AS
#         SELECT entity_id, business_name, business_address, country
#         FROM read_csv('{cfg["test_path"]}', delim='\t', header=True, auto_detect=True);
#     """)
#     n = con.execute(f"SELECT COUNT(*) FROM {s_table}").fetchone()[0]
#     print(f"  loaded {n:,} {source_name} records")
 
#     print("  building candidates (SNM, multi-pass)...")
#     cand_table = f"candidates_{source_name}"
#     build_candidates(con, s1_table, s_table, cand_table)
 
#     print("  writing candidate chunks to disk...")
#     chunk_dir = os.path.join(WORK_DIR, f"candidates_{source_name}")
#     chunk_files = write_candidate_chunks(con, cand_table, s1_table, s_table, chunk_dir)
 
#     if not chunk_files:
#         print("  no candidates generated, skipping inference")
#         return f"results_{source_name}"
 
#     print(f"  scoring {len(chunk_files):,} chunks across {N_WORKERS} workers...")
#     t0 = time.time()
#     result_files = []
#     with ProcessPoolExecutor(max_workers=N_WORKERS,
#                               initializer=_init_worker,
#                               initargs=(model_path,)) as ex:
#         for i, path in enumerate(ex.map(_process_chunk_file, chunk_files, chunksize=4)):
#             result_files.append(path)
#             if (i + 1) % 50 == 0:
#                 print(f"    {i + 1:,}/{len(chunk_files):,} chunks scored "
#                       f"({time.time() - t0:.1f}s elapsed)")
#     print(f"  scoring done in {time.time() - t0:.1f}s")
 
#     result_dir = os.path.join(WORK_DIR, f"results_{source_name}")
#     if os.path.exists(result_dir):
#         shutil.rmtree(result_dir)
#     os.makedirs(result_dir, exist_ok=True)
#     for i, f in enumerate(result_files):
#         shutil.move(f, os.path.join(result_dir, f"part_{i}.parquet"))
 
#     return result_dir





def run_source(con, source_name, cfg, s1_table):
    print(f"\n=== Source: {source_name} ===")
    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        fallback = SOURCES["s2"]["model_path"]
        print(f"  [!] {model_path} not found, falling back to {fallback}")
        model_path = fallback

    s_table = f"test_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TABLE {s_table} AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{cfg["test_path"]}', delim='\t', header=True, auto_detect=True);
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {s_table}").fetchone()[0]
    print(f"  loaded {n:,} {source_name} records")

    print("  building candidates (SNM, multi-pass)...")
    cand_table = f"candidates_{source_name}"
    build_candidates(con, s1_table, s_table, cand_table)

    print("  writing candidate chunks to disk...")
    chunk_dir = os.path.join(WORK_DIR, f"candidates_{source_name}")
    chunk_files = write_candidate_chunks(con, cand_table, s1_table, s_table, chunk_dir)

    if not chunk_files:
        print("  no candidates generated, skipping inference")
        return f"results_{source_name}"

    # --- FIX ADDED HERE ---
    # Pre-create the result directory so workers don't fail when saving parquet chunks
    result_dir = os.path.join(WORK_DIR, f"results_{source_name}")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    # ----------------------

    print(f"  scoring {len(chunk_files):,} chunks across {N_WORKERS} workers...")
    t0 = time.time()
    result_files = []
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                              initializer=_init_worker,
                              initargs=(model_path,)) as ex:
        for i, path in enumerate(ex.map(_process_chunk_file, chunk_files, chunksize=4)):
            result_files.append(path)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1:,}/{len(chunk_files):,} chunks scored "
                      f"({time.time() - t0:.1f}s elapsed)")
    print(f"  scoring done in {time.time() - t0:.1f}s")

    return result_dir
 
 
def rank_and_cap(con, result_glob, max_matches, out_table):
    """Top-K by probability per s1_id -- fixes the 'first seen' bug."""
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
    # Point DuckDB's spill directory somewhere with real free space if the
    # default drive is tight -- uncomment and adjust:
    # con.execute("PRAGMA temp_directory='D:/duckdb_tmp';")
 
    limit_clause = f"LIMIT {DRY_RUN_S1_LIMIT}" if DRY_RUN_S1_LIMIT else ""
    if DRY_RUN_S1_LIMIT:
        print(f"*** DRY RUN: capping S1 to {DRY_RUN_S1_LIMIT:,} rows. "
              f"Set DRY_RUN_S1_LIMIT = None for the full run. ***")
 
    con.execute(f"""
        CREATE OR REPLACE TABLE test_s1 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{S1_PATH}', delim='\t', header=True, auto_detect=True)
        {limit_clause};
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
 
    print("\nAssembling final submission (S2 + S3 merged, ranked by probability)...")
    union_parts = []
    for source_name, table in ranked_tables.items():
        if table:
            union_parts.append(f"SELECT s1_id, s2_id, prob FROM {table}")
    if not union_parts:
        raise RuntimeError("No matches found for any source -- check blocking windows / threshold.")
 
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
    print(f"\nSUCCESS in {time.time() - t0:.2f}s")
    print(f"Total S1 entities: {len(submission):,}")
    print(f"S1 entities with >=1 match: {n_matched:,}")
 
 
if __name__ == "__main__":
    main()
 




































# # fast_inference.py
# import polars as pl
# import duckdb
# import numpy as np
# import lightgbm as lgb
# from rapidfuzz import fuzz
# from concurrent.futures import ProcessPoolExecutor
# import re
# import os
# import time

# # Top-level worker function for multi-threaded feature extraction
# def process_chunk_worker(args):
#     s1_n, s1_a, s2_n, s2_a = args
#     s1_n, s1_a = s1_n or "", s1_a or ""
#     s2_n, s2_a = s2_n or "", s2_a or ""
    
#     n_ratio = fuzz.ratio(s1_n, s2_n)
#     n_partial = fuzz.partial_ratio(s1_n, s2_n)
#     n_sort = fuzz.token_sort_ratio(s1_n, s2_n)
#     n_set = fuzz.token_set_ratio(s1_n, s2_n)
    
#     a_ratio = fuzz.ratio(s1_a, s2_a)
#     a_set = fuzz.token_set_ratio(s1_a, s2_a)
    
#     s1_nums = set(re.findall(r'\d+', s1_a))
#     s2_nums = set(re.findall(r'\d+', s2_a))
#     num_match = 1.0 if s1_nums and s2_nums and (s1_nums & s2_nums) else 0.0
    
#     len_diff_n = abs(len(s1_n) - len(s2_n))
#     len_diff_a = abs(len(s1_a) - len(s2_a))
#     missing_addr = 1.0 if not s1_a or not s2_a else 0.0
    
#     return [n_ratio, n_partial, n_sort, n_set, a_ratio, a_set, num_match, len_diff_n, len_diff_a, missing_addr]


# if __name__ == '__main__':
#     print("--- PARALLEL INFERENCE ENGINE (FIRST-WORD + LENGTH BOUND BLOCKING) ---")
#     t0 = time.time()

#     # 1. Load LightGBM Model
#     model_path = "matcher_s2_model.txt"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError("Model file not found! Run train_model.py first.")

#     model = lgb.Booster(model_file=model_path)
#     print("Loaded LightGBM matcher model successfully.")

#     # 2. Connect DuckDB and Load Test Sets with First-Word Tokens and Lengths
#     print("\nLoading test datasets into DuckDB with length bounds...")
#     con = duckdb.connect()
#     con.execute("SET memory_limit = '6GB';")

#     con.execute("""
#         CREATE TABLE test_s1 AS 
#         SELECT entity_id, business_name, business_address, country,
#                LENGTH(COALESCE(business_name, '')) AS name_len,
#                STR_SPLIT(TRIM(LOWER(REGEXP_REPLACE(COALESCE(business_name, ''), '[^a-zA-Z0-9 ]', '', 'g'))), ' ')[1] AS block_key
#         FROM read_csv('dataset/test/test_source1.tsv', delim='\t', header=True, auto_detect=True);
#     """)

#     con.execute("""
#         CREATE TABLE test_s2 AS 
#         SELECT entity_id, business_name, business_address, country,
#                LENGTH(COALESCE(business_name, '')) AS name_len,
#                STR_SPLIT(TRIM(LOWER(REGEXP_REPLACE(COALESCE(business_name, ''), '[^a-zA-Z0-9 ]', '', 'g'))), ' ')[1] AS block_key
#         FROM read_csv('dataset/test/test_source2.tsv', delim='\t', header=True, auto_detect=True);
#     """)

#     s1_count = con.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]
#     s2_count = con.execute("SELECT COUNT(*) FROM test_s2").fetchone()[0]
#     print(f"Loaded {s1_count:,} S1 and {s2_count:,} S2 records in {time.time() - t0:.2f}s!")

#     # 3. Join on Country, First-Word Token, and String Length Bounds (<= 15 chars)
#     print("\nRunning SQL Token-Block Matching (Strict First-Word + Length Filter)...")
#     con.execute("""
#         CREATE TABLE matches_raw AS
#         SELECT 
#             s1.entity_id AS s1_id,
#             s2.entity_id AS s2_id,
#             s1.business_name AS s1_name,
#             s1.business_address AS s1_addr,
#             s2.business_name AS s2_name,
#             s2.business_address AS s2_addr
#         FROM test_s1 s1
#         JOIN test_s2 s2 
#           ON s1.country = s2.country 
#          AND s1.block_key = s2.block_key
#         WHERE s1.block_key IS NOT NULL 
#           AND s1.block_key NOT IN ('', 'the', 'a', 'inc', 'llc', 'corp', 'ltd', 'co')
#           AND ABS(s1.name_len - s2.name_len) <= 15;
#     """)

#     matched_pairs_count = con.execute("SELECT COUNT(*) FROM matches_raw").fetchone()[0]
#     print(f"Optimized blocking generated {matched_pairs_count:,} candidate pairs to evaluate.")

#     # 4. Extract numpy arrays for multi-core processing
#     print("\nEvaluating candidate pairs using LightGBM (Multi-Core Pipeline)...")
#     raw_df = con.execute("SELECT * FROM matches_raw").pl()

#     s1_ids = raw_df["s1_id"].to_numpy()
#     s2_ids = raw_df["s2_id"].to_numpy()
#     s1_names = raw_df["s1_name"].to_numpy()
#     s1_addrs = raw_df["s1_addr"].to_numpy()
#     s2_names = raw_df["s2_name"].to_numpy()
#     s2_addrs = raw_df["s2_addr"].to_numpy()

#     total_rows = len(raw_df)
#     results = {}
#     PROB_THRESHOLD = 0.50
#     MAX_S2_MATCHES = 5
#     BATCH_SIZE = 100000

#     # Multi-core execution across all CPU threads
#     with ProcessPoolExecutor() as executor:
#         for b_start in range(0, total_rows, BATCH_SIZE):
#             b_end = min(b_start + BATCH_SIZE, total_rows)
#             print(f"--> Processing batch {b_start:,} to {b_end:,} / {total_rows:,}...")
            
#             chunk_args = list(zip(
#                 s1_names[b_start:b_end],
#                 s1_addrs[b_start:b_end],
#                 s2_names[b_start:b_end],
#                 s2_addrs[b_start:b_end]
#             ))
            
#             X_batch = list(executor.map(process_chunk_worker, chunk_args, chunksize=2000))
            
#             if X_batch:
#                 probs = model.predict(np.array(X_batch))
#                 b_s1_ids = s1_ids[b_start:b_end]
#                 b_s2_ids = s2_ids[b_start:b_end]
                
#                 for idx in range(len(probs)):
#                     if probs[idx] >= PROB_THRESHOLD:
#                         s1_id = b_s1_ids[idx]
#                         s2_id = b_s2_ids[idx]
                        
#                         if s1_id not in results:
#                             results[s1_id] = []
#                         if len(results[s1_id]) < MAX_S2_MATCHES:
#                             results[s1_id].append(s2_id)

#     # 5. Format Submission Output
#     print("\nGenerating submission.tsv file...")
#     all_s1_ids = con.execute("SELECT entity_id FROM test_s1").pl()["entity_id"].to_numpy()

#     out_matches = []
#     for s1_id in all_s1_ids:
#         matches = results.get(s1_id, [])
#         out_matches.append(",".join(matches) if matches else "")

#     submission_df = pl.DataFrame({
#         "source1_entity_id": all_s1_ids,
#         "matched_entity_ids": out_matches
#     })

#     submission_df.write_csv("submission.tsv", separator="\t")

#     print(f"\nSUCCESS! Submission file generated in {time.time() - t0:.2f} seconds.")
#     print(f"Total test entities in output: {len(submission_df):,}")
#     print(f"Total entities with matches: {len([m for m in out_matches if m]):,}")