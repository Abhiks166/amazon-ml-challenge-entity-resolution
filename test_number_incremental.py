"""Measure filtered address-number recovery only for four-pass GT misses.

This diagnostic reuses the exact SNM ranks and pass predicates from
analyze_blocking_misses.py / fast_inference.py. It does not construct a global
S1-to-source candidate table: number joins are limited to GT misses or their
distinct S1 IDs for per-S1 cost statistics.
"""

from pathlib import Path

import duckdb

from analyze_blocking_misses import (
    ALL_PASSES,
    DB_MEMORY_LIMIT,
    DB_THREADS,
    create_pass_status,
    create_ranked_records,
    register_transliteration_udf,
)


DATASET_DIR = Path(
    "/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset/train"
)
CUTOFFS = (25, 50, 100, 250, 500)


def dataset_path(filename: str) -> Path:
    path = DATASET_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Expected Kaggle training file was not found: {path}")
    return path


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def create_ground_truth_pairs(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ground_truth_pairs AS
        SELECT
            source1_entity_id AS s1_id,
            trim(match_id) AS source_entity_id,
            trim(match_id) AS source_id
        FROM train_ground_truth,
             UNNEST(str_split(coalesce(matched_entity_ids, ''), ',')) AS matches(match_id)
        WHERE trim(match_id) <> '';
    """)


def create_four_pass_misses(
    con: duckdb.DuckDBPyConnection, source_name: str, source_table: str
) -> tuple[str, int, int]:
    """Evaluate GT pairs against the exact four existing pre-cap passes."""
    status_tables: list[str] = []
    for pass_name, config in ALL_PASSES.items():
        ranked_table = create_ranked_records(
            con, source_table, source_name, pass_name, config["window"]
        )
        status_table = create_pass_status(con, source_name, ranked_table, pass_name)
        status_tables.append(status_table)
        con.execute(f"DROP TABLE IF EXISTS {ranked_table};")

    source_prefix = source_name.upper() + "-"
    misses_table = f"four_pass_misses_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {misses_table} AS
        SELECT name_pass.s1_id, name_pass.source_entity_id AS source_id
        FROM {status_tables[0]} name_pass
        JOIN {status_tables[1]} token_pass
          ON token_pass.s1_id = name_pass.s1_id
         AND token_pass.source_entity_id = name_pass.source_entity_id
        JOIN {status_tables[2]} address_pass
          ON address_pass.s1_id = name_pass.s1_id
         AND address_pass.source_entity_id = name_pass.source_entity_id
        JOIN {status_tables[3]} transliterated_pass
          ON transliterated_pass.s1_id = name_pass.s1_id
         AND transliterated_pass.source_entity_id = name_pass.source_entity_id
        WHERE NOT name_pass.recovered
          AND NOT token_pass.recovered
          AND NOT address_pass.recovered
          AND NOT transliterated_pass.recovered;
    """)
    for status_table in status_tables:
        con.execute(f"DROP TABLE IF EXISTS {status_table};")

    total = con.execute(
        f"SELECT COUNT(*) FROM ground_truth_pairs WHERE source_id LIKE '{source_prefix}%';"
    ).fetchone()[0]
    misses = con.execute(f"SELECT COUNT(*) FROM {misses_table};").fetchone()[0]
    return misses_table, total, misses


def create_number_index(con: duckdb.DuckDBPyConnection, source_table: str, index_table: str) -> str:
    """One unique (entity_id, country, digit sequence) posting per record."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {index_table} AS
        SELECT DISTINCT entity_id, country, number AS block_key
        FROM {source_table},
             UNNEST(regexp_extract_all(coalesce(business_address, ''), '\\d+')) AS numbers(number)
        WHERE number <> '';
    """)
    return index_table


def create_target_frequency(con: duckdb.DuckDBPyConnection, index_table: str, frequency_table: str) -> str:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {frequency_table} AS
        SELECT country, block_key, COUNT(*) AS postings
        FROM {index_table}
        GROUP BY country, block_key;
    """)
    return frequency_table


def create_pair_minimum_posting(
    con: duckdb.DuckDBPyConnection,
    misses_table: str,
    source_name: str,
    source_numbers: str,
    source_frequency: str,
) -> str:
    """For each four-pass-missed true pair, retain its cheapest shared number."""
    table = f"pair_minimum_posting_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT misses.s1_id, misses.source_id, MIN(frequency.postings) AS min_postings
        FROM {misses_table} misses
        JOIN s1_number_index s1_number ON s1_number.entity_id = misses.s1_id
        JOIN {source_numbers} source_number
          ON source_number.entity_id = misses.source_id
         AND source_number.country = s1_number.country
         AND source_number.block_key = s1_number.block_key
        JOIN {source_frequency} frequency
          ON frequency.country = source_number.country
         AND frequency.block_key = source_number.block_key
        GROUP BY misses.s1_id, misses.source_id;
    """)
    return table


def number_cost_metrics(
    con: duckdb.DuckDBPyConnection,
    missed_s1: str,
    source_numbers: str,
    source_frequency: str,
    cutoff: int,
) -> tuple[float, float, float, float, int, int]:
    """Compute exact unique target candidates per current-miss S1, without a pair table."""
    # COUNT(DISTINCT source entity) removes pairs that share multiple usable
    # numbers. The only materialized relation is one aggregate row per missed
    # S1 entity, never the full number-block candidate-pair relation.
    row = con.execute(f"""
        WITH candidate_counts AS (
            SELECT missed.entity_id AS s1_id, COUNT(DISTINCT source_number.entity_id) AS candidates
            FROM {missed_s1} missed
            JOIN s1_number_index s1_number ON s1_number.entity_id = missed.entity_id
            JOIN {source_numbers} source_number
              ON source_number.country = s1_number.country
             AND source_number.block_key = s1_number.block_key
            JOIN {source_frequency} frequency
              ON frequency.country = source_number.country
             AND frequency.block_key = source_number.block_key
            WHERE frequency.postings <= {cutoff}
            GROUP BY missed.entity_id
        )
        SELECT
            AVG(candidates),
            quantile_cont(candidates, 0.50),
            quantile_cont(candidates, 0.95),
            quantile_cont(candidates, 0.99),
            MAX(candidates),
            COUNT(*)
        FROM candidate_counts;
    """).fetchone()
    return tuple(value or 0 for value in row)


def report_source(con: duckdb.DuckDBPyConnection, source_name: str, source_table: str) -> None:
    prefix = source_name.upper() + "-"
    misses_table, total_pairs, current_misses = create_four_pass_misses(
        con, source_name, source_table
    )
    current_hits = total_pairs - current_misses
    missed_s1 = f"missed_s1_{source_name}"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {missed_s1} AS SELECT DISTINCT s1_id AS entity_id FROM {misses_table};")

    source_numbers = create_number_index(con, source_table, f"{source_name}_number_index")
    source_frequency = create_target_frequency(con, source_numbers, f"{source_name}_number_frequency")
    pair_minimum = create_pair_minimum_posting(
        con, misses_table, source_name, source_numbers, source_frequency
    )

    print(f"\n--- S1 -> {source_name.upper()} filtered number increment (PRE-CAP) ---")
    print(f"Total GT pairs: {total_pairs:,}")
    print(f"Current four-pass hits: {current_hits:,}")
    print(f"Current four-pass misses: {current_misses:,}")
    print(f"Distinct S1 entities requiring number recovery: {con.execute(f'SELECT COUNT(*) FROM {missed_s1}').fetchone()[0]:,}")
    print("cutoff | incremental recovered | % of current misses | overall recall | mean/p50/p95/p99/max candidates per missed S1 | S1 with candidates")

    for cutoff in CUTOFFS:
        recovered = con.execute(
            f"SELECT COUNT(*) FROM {pair_minimum} WHERE min_postings <= {cutoff};"
        ).fetchone()[0]
        incremental_recall = recovered / current_misses if current_misses else 0.0
        overall_recall = (current_hits + recovered) / total_pairs if total_pairs else 0.0
        mean, p50, p95, p99, maximum, s1_with_candidates = number_cost_metrics(
            con, missed_s1, source_numbers, source_frequency, cutoff
        )
        print(
            f"{cutoff:>6,} | {recovered:>21,} | {incremental_recall:>18.4%} | "
            f"{overall_recall:>14.4%} | {mean:.1f}/{p50:.0f}/{p95:.0f}/{p99:.0f}/{maximum:.0f} | "
            f"{s1_with_candidates:,}"
        )

    for table in (misses_table, missed_s1, source_numbers, source_frequency, pair_minimum):
        con.execute(f"DROP TABLE IF EXISTS {table};")


def main() -> None:
    paths = {
        filename: dataset_path(filename)
        for filename in (
            "train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"
        )
    }
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DB_MEMORY_LIMIT}';")
    con.execute(f"SET threads = {DB_THREADS};")
    con.execute("SET preserve_insertion_order = false;")
    register_transliteration_udf(con)
    con.execute(f"""
        CREATE OR REPLACE TABLE train_s1 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(paths['train_source1.tsv'])}', delim='\t', header=True, auto_detect=True);
        CREATE OR REPLACE TABLE train_s2 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(paths['train_source2.tsv'])}', delim='\t', header=True, auto_detect=True);
        CREATE OR REPLACE TABLE train_s3 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(paths['train_source3.tsv'])}', delim='\t', header=True, auto_detect=True);
        CREATE OR REPLACE TABLE train_ground_truth AS
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{sql_path(paths['train_ground_truth.tsv'])}', delim='\t', header=True, auto_detect=True);
    """)
    create_ground_truth_pairs(con)
    create_number_index(con, "train_s1", "s1_number_index")
    print("--- FILTERED ADDRESS-NUMBER INCREMENTAL RECOVERY (PRE-CAP) ---")
    report_source(con, "s2", "train_s2")
    report_source(con, "s3", "train_s3")
    con.execute("DROP TABLE s1_number_index;")


if __name__ == "__main__":
    main()
