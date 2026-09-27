"""Measure practical candidate volume for additive blocking strategies.

This diagnostic does not modify production code. It uses real posting-list
joins (not GT overlap alone), never applies the production 40-candidate cap,
and reports pre-cap recall separately from raw candidate volume.
"""

import platform
import resource
import time

import duckdb

from analyze_blocking_misses import (
    ALL_PASSES,
    DB_MEMORY_LIMIT,
    DB_THREADS,
    create_ranked_records,
    locate_training_file,
    normalized_sql,
    register_transliteration_udf,
    sql_path,
)


CHAR_N = 3
CHAR_FILTER_MAX_POSTINGS = 5_000
NUMBER_FILTER_MAX_POSTINGS = 2_000
LARGE_CANDIDATE_SET = 1_000


def create_ground_truth_pairs(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE ground_truth_pairs AS
        SELECT source1_entity_id AS s1_id, trim(match_id) AS source_id
        FROM train_ground_truth,
             UNNEST(str_split(coalesce(matched_entity_ids, ''), ',')) AS matches(match_id)
        WHERE trim(match_id) <> '';
    """)


def bucket_pair_sql(ranked_table: str) -> str:
    """The two pre-cap bucket joins in fast_inference.py's build_candidates()."""
    return f"""
        SELECT
            CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END AS s1_id,
            CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END AS source_id
        FROM {ranked_table} a
        JOIN {ranked_table} b
          ON a.country = b.country
         AND a.bucket = b.bucket
         AND a.src <> b.src
        UNION ALL
        SELECT
            CASE WHEN a.src = 'A' THEN a.entity_id ELSE b.entity_id END AS s1_id,
            CASE WHEN a.src = 'A' THEN b.entity_id ELSE a.entity_id END AS source_id
        FROM {ranked_table} a
        JOIN {ranked_table} b
          ON a.country = b.country
         AND a.bucket + 1 = b.bucket
         AND a.src <> b.src
    """


def build_baseline_candidates(con: duckdb.DuckDBPyConnection, source_name: str, source_table: str) -> str:
    """Materialize the exact four current passes, except intentionally pre-cap."""
    parts = []
    for pass_name, config in ALL_PASSES.items():
        ranked = create_ranked_records(con, source_table, source_name, pass_name, config["window"])
        parts.append(bucket_pair_sql(ranked))
    table = f"baseline_{source_name}"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT DISTINCT * FROM ({' UNION ALL '.join(parts)});")
    return table


def create_char_index(con: duckdb.DuckDBPyConnection, table: str, index_table: str) -> None:
    norm_name = normalized_sql("business_name")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {index_table} AS
        SELECT DISTINCT entity_id, country, substr(normalized_name, position, {CHAR_N}) AS block_key
        FROM (
            SELECT entity_id, country, {norm_name} AS normalized_name
            FROM {table}
        ) names,
        UNNEST(generate_series(1, length(normalized_name) - {CHAR_N} + 1)) AS positions(position)
        WHERE length(normalized_name) >= {CHAR_N};
    """)


def create_number_index(con: duckdb.DuckDBPyConnection, table: str, index_table: str) -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {index_table} AS
        SELECT DISTINCT entity_id, country, number AS block_key
        FROM {table},
        UNNEST(regexp_extract_all(coalesce(business_address, ''), '\\d+')) AS numbers(number)
        WHERE number <> '';
    """)


def print_frequency_distribution(con: duckdb.DuckDBPyConnection, source_index: str, label: str, threshold: int) -> None:
    """Report posting-list sizes before creating a candidate-pair join."""
    row = con.execute(f"""
        WITH frequencies AS (
            SELECT country, block_key, COUNT(*) AS postings
            FROM {source_index}
            GROUP BY country, block_key
        )
        SELECT
            COUNT(*),
            quantile_cont(postings, 0.50), quantile_cont(postings, 0.90),
            quantile_cont(postings, 0.95), quantile_cont(postings, 0.99), MAX(postings),
            COUNT(*) FILTER (WHERE postings > {threshold}),
            SUM(postings) FILTER (WHERE postings > {threshold})
        FROM frequencies;
    """).fetchone()
    keys, p50, p90, p95, p99, maximum, over, rows_over = row
    print(
        f"  {label} source posting lists: keys={keys:,}, p50={p50:.0f}, p90={p90:.0f}, "
        f"p95={p95:.0f}, p99={p99:.0f}, max={maximum:,}; "
        f"over {threshold:,}: {over:,} keys / {rows_over or 0:,} postings"
    )


def build_index_candidates(
    con: duckdb.DuckDBPyConnection,
    s1_index: str,
    source_index: str,
    table: str,
    max_source_postings: int | None,
) -> str:
    """Materialize actual unique pairs from a same-country inverted index."""
    filter_sql = ""
    if max_source_postings is not None:
        filter_sql = f"WHERE source_frequency.postings <= {max_source_postings}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        WITH source_frequency AS (
            SELECT country, block_key, COUNT(*) AS postings
            FROM {source_index}
            GROUP BY country, block_key
        )
        SELECT DISTINCT s1.entity_id AS s1_id, source.entity_id AS source_id
        FROM {s1_index} s1
        JOIN {source_index} source
          ON s1.country = source.country
         AND s1.block_key = source.block_key
        JOIN source_frequency
          ON source_frequency.country = source.country
         AND source_frequency.block_key = source.block_key
        {filter_sql};
    """)
    return table


def measure_candidates(
    con: duckdb.DuckDBPyConnection,
    label: str,
    candidates: str,
    source_prefix: str,
    started_at: float,
    eligible_s1: str = "train_s1",
    baseline: str | None = None,
) -> dict[str, float | int]:
    """Report recall and actual, deduplicated candidate-pair distribution."""
    metrics = con.execute(f"""
        WITH true_pairs AS (
            SELECT s1_id, source_id
            FROM ground_truth_pairs
            WHERE source_id LIKE '{source_prefix}%'
        ),
        per_s1 AS (
            SELECT s1_id, COUNT(*) AS candidates
            FROM {candidates}
            GROUP BY s1_id
        )
        SELECT
            (SELECT COUNT(*) FROM true_pairs),
            (SELECT COUNT(*) FROM true_pairs gt JOIN {candidates} c
                ON c.s1_id = gt.s1_id AND c.source_id = gt.source_id),
            (SELECT COUNT(*) FROM {candidates}),
            (SELECT AVG(candidates) FROM per_s1),
            (SELECT quantile_cont(candidates, 0.50) FROM per_s1),
            (SELECT quantile_cont(candidates, 0.90) FROM per_s1),
            (SELECT quantile_cont(candidates, 0.95) FROM per_s1),
            (SELECT quantile_cont(candidates, 0.99) FROM per_s1),
            (SELECT MAX(candidates) FROM per_s1),
            (SELECT COUNT(*) FROM {eligible_s1} s LEFT JOIN per_s1 p ON p.s1_id = s.entity_id
                WHERE p.s1_id IS NULL),
            (SELECT COUNT(*) FROM per_s1 WHERE candidates > {LARGE_CANDIDATE_SET});
    """).fetchone()
    total, recovered, volume, mean, median, p90, p95, p99, maximum, zero, large = metrics
    additional = 0
    if baseline is not None:
        additional = con.execute(f"""
            SELECT COUNT(*)
            FROM ground_truth_pairs gt
            JOIN {candidates} c ON c.s1_id = gt.s1_id AND c.source_id = gt.source_id
            LEFT JOIN {baseline} b ON b.s1_id = gt.s1_id AND b.source_id = gt.source_id
            WHERE gt.source_id LIKE '{source_prefix}%' AND b.s1_id IS NULL;
        """).fetchone()[0]
    elapsed = time.perf_counter() - started_at
    recall = recovered / total if total else 0.0
    result = {
        "label": label, "total": total, "recovered": recovered, "recall": recall,
        "volume": volume, "mean": mean or 0, "median": median or 0, "p90": p90 or 0,
        "p95": p95 or 0, "p99": p99 or 0, "max": maximum or 0, "zero": zero,
        "large": large, "additional": additional, "runtime": elapsed,
    }
    print(
        f"  {label}: recall={recall:.4%} ({recovered:,}/{total:,}), candidates={volume:,}, "
        f"per-S1 mean/p50/p90/p95/p99/max={result['mean']:.1f}/{result['median']:.0f}/"
        f"{result['p90']:.0f}/{result['p95']:.0f}/{result['p99']:.0f}/{result['max']:.0f}, "
        f"zero={zero:,}, >{LARGE_CANDIDATE_SET:,}={large:,}, runtime={elapsed:.1f}s"
    )
    return result


def union_candidates(con: duckdb.DuckDBPyConnection, table: str, *inputs: str) -> str:
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS " + " UNION ".join(f"SELECT * FROM {item}" for item in inputs) + ";")
    return table


def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def run_source(con: duckdb.DuckDBPyConnection, source_name: str, source_table: str) -> None:
    prefix = source_name.upper() + "-"
    print(f"\n=== S1 -> {source_name.upper()} ===")
    started = time.perf_counter()
    baseline = build_baseline_candidates(con, source_name, source_table)
    results = [measure_candidates(con, "A current four-pass baseline", baseline, prefix, started)]

    create_char_index(con, "train_s1", "s1_char_index")
    create_char_index(con, source_table, "source_char_index")
    print_frequency_distribution(con, "source_char_index", "char-3", CHAR_FILTER_MAX_POSTINGS)
    char_naive = build_index_candidates(con, "s1_char_index", "source_char_index", f"char_naive_{source_name}", None)
    results.append(measure_candidates(con, "B1 char-3 naive", char_naive, prefix, started))
    con.execute(f"DROP TABLE {char_naive};")
    char_filtered = build_index_candidates(con, "s1_char_index", "source_char_index", f"char_filtered_{source_name}", CHAR_FILTER_MAX_POSTINGS)
    results.append(measure_candidates(con, "B2 char-3 filtered", char_filtered, prefix, started))

    create_number_index(con, "train_s1", "s1_number_index")
    create_number_index(con, source_table, "source_number_index")
    print_frequency_distribution(con, "source_number_index", "address number", NUMBER_FILTER_MAX_POSTINGS)
    number_naive = build_index_candidates(con, "s1_number_index", "source_number_index", f"number_naive_{source_name}", None)
    measure_candidates(con, "C1 address-number naive", number_naive, prefix, started)
    con.execute(f"DROP TABLE {number_naive};")
    number_filtered = build_index_candidates(con, "s1_number_index", "source_number_index", f"number_filtered_{source_name}", NUMBER_FILTER_MAX_POSTINGS)
    results.append(measure_candidates(con, "C2 address-number filtered", number_filtered, prefix, started))

    baseline_char = union_candidates(con, f"baseline_char_{source_name}", baseline, char_filtered)
    results.append(measure_candidates(con, "D baseline + char-3 filtered", baseline_char, prefix, started, baseline=baseline))
    baseline_number = union_candidates(con, f"baseline_number_{source_name}", baseline, number_filtered)
    results.append(measure_candidates(con, "E baseline + address-number filtered", baseline_number, prefix, started, baseline=baseline))
    all_strategies = union_candidates(con, f"all_strategies_{source_name}", baseline, char_filtered, number_filtered)
    results.append(measure_candidates(con, "F baseline + both filtered", all_strategies, prefix, started, baseline=baseline))

    recovery_s1 = f"baseline_zero_s1_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {recovery_s1} AS
        SELECT s.entity_id
        FROM train_s1 s
        LEFT JOIN (SELECT DISTINCT s1_id FROM {baseline}) b ON b.s1_id = s.entity_id
        WHERE b.s1_id IS NULL;
    """)
    recovery_char = f"recovery_char_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {recovery_char} AS
        SELECT c.* FROM {char_filtered} c JOIN {recovery_s1} z ON z.entity_id = c.s1_id;
    """)
    measure_candidates(con, "char-3 recovery-only (baseline-zero S1 only)", recovery_char, prefix, started, recovery_s1, baseline)

    print(f"  Peak process memory observed: {peak_memory_mb():,.1f} MB")
    print("  Concise comparison (pre-cap):")
    print("    strategy | recall | candidates | mean/S1 | p99 | max | zero S1 | additional GT")
    for result in results:
        print(
            f"    {result['label']} | {result['recall']:.4%} | {result['volume']:,} | "
            f"{result['mean']:.1f} | {result['p99']:.0f} | {result['max']:.0f} | "
            f"{result['zero']:,} | {result['additional']:,}"
        )
    print(
        "  Recommendation rule: use a global additive pass only if its measured "
        "recall gain justifies its candidate-volume tail; otherwise prefer the "
        "recovery-only result for S1 entities with no baseline candidates."
    )


def main() -> None:
    paths = {name: locate_training_file(name) for name in (
        "train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"
    )}
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DB_MEMORY_LIMIT}';")
    con.execute(f"SET threads = {DB_THREADS};")
    con.execute("SET preserve_insertion_order = false;")
    register_transliteration_udf(con)
    con.execute(f"""
        CREATE OR REPLACE TABLE train_s1 AS SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(paths['train_source1.tsv'])}', delim='\t', header=True, auto_detect=True);
        CREATE OR REPLACE TABLE train_s2 AS SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(paths['train_source2.tsv'])}', delim='\t', header=True, auto_detect=True);
        CREATE OR REPLACE TABLE train_s3 AS SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(paths['train_source3.tsv'])}', delim='\t', header=True, auto_detect=True);
        CREATE OR REPLACE TABLE train_ground_truth AS SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{sql_path(paths['train_ground_truth.tsv'])}', delim='\t', header=True, auto_detect=True);
    """)
    create_ground_truth_pairs(con)
    print("--- BLOCKING STRATEGY FEASIBILITY (ALL RESULTS ARE PRE-CAP) ---")
    run_source(con, "s2", "train_s2")
    run_source(con, "s3", "train_s3")


if __name__ == "__main__":
    main()
