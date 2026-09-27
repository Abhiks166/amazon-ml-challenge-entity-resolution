"""Diagnose why ground-truth pairs are missed by fast_inference.py's blocker.

Run later with:
    python analyze_blocking_misses.py

The script evaluates only ground-truth pairs against the same ranked buckets
used by fast_inference.py. It deliberately does not materialize the full
candidate-pair join and does not alter the blocker.
"""

from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import duckdb

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError as error:
    raise ImportError(
        "analyze_blocking_misses.py requires indic-transliteration. "
        "Install it with: pip install indic-transliteration"
    ) from error


# Exact blocking configuration from fast_inference.py.
DB_MEMORY_LIMIT = "4GB"
DB_THREADS = 4
SNM_PASSES = {
    "name": {"window": 25},
    "token_sorted": {"window": 15},
    "address": {"window": 15},
}
TRANSLITERATED_NAME_PASS = {"transliterated_name": {"window": 25}}
ALL_PASSES = {**SNM_PASSES, **TRANSLITERATED_NAME_PASS}
SAMPLE_SIZE = 50

# These ranges cover the Indic scripts targeted by this experiment. Individual
# script spans are transliterated so mixed Latin/Indic business names retain
# their existing Latin text.
INDIC_SCRIPT_SCHEMES = (
    (re.compile(r"[\u0900-\u097F]+"), sanscript.DEVANAGARI),
    (re.compile(r"[\u0980-\u09FF]+"), sanscript.BENGALI),
    (re.compile(r"[\u0A80-\u0AFF]+"), sanscript.GUJARATI),
    (re.compile(r"[\u0B80-\u0BFF]+"), sanscript.TAMIL),
    (re.compile(r"[\u0C00-\u0C7F]+"), sanscript.TELUGU),
    (re.compile(r"[\u0C80-\u0CFF]+"), sanscript.KANNADA),
    (re.compile(r"[\u0D00-\u0D7F]+"), sanscript.MALAYALAM),
)


def locate_training_file(filename: str) -> Path:
    """Find one required training TSV below the repository directory."""
    repository_dir = Path(__file__).resolve().parent
    common_locations = (
        Path("/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset/train") / filename,
        repository_dir / "dataset" / "train" / filename,
        repository_dir / "dataset" / filename,
        repository_dir / "data" / "train" / filename,
        repository_dir / "data" / filename,
        repository_dir / filename,
    )
    for path in common_locations:
        if path.is_file():
            return path.resolve()

    matches = sorted(repository_dir.rglob(filename))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        locations = "\n  ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Found multiple files named {filename}. Select one by updating "
            f"locate_training_file():\n  {locations}"
        )

    checked = "\n  ".join(str(path) for path in common_locations)
    raise FileNotFoundError(
        f"Could not locate {filename}. Searched recursively below "
        f"{repository_dir}. Common locations checked:\n  {checked}"
    )


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def normalize_transliterated_name(value: str) -> str:
    """Normalize transliteration output after, not instead of, the existing key."""
    decomposed = unicodedata.normalize("NFKD", value.lower())
    without_diacritics = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9 ]", "", without_diacritics).strip()


@lru_cache(maxsize=500_000)
def transliterated_name(value: str) -> str:
    """Convert supported Indic-script spans to IAST, then normalize the result."""
    text = value or ""
    for pattern, source_scheme in INDIC_SCRIPT_SCHEMES:
        text = pattern.sub(
            lambda match, scheme=source_scheme: transliterate(
                match.group(0), scheme, sanscript.IAST
            ),
            text,
        )
    return normalize_transliterated_name(text)


def register_transliteration_udf(con: duckdb.DuckDBPyConnection) -> None:
    """Expose the cached helper to DuckDB while ranked rows are constructed."""
    con.create_function(
        "transliterated_business_name",
        transliterated_name,
        return_type="VARCHAR",
        null_handling="special",
    )


def sort_key_sql(kind: str, name_col: str = "business_name", addr_col: str = "business_address") -> str:
    """Return the exact sort-key expression from fast_inference.py."""
    norm_name = (
        f"trim(lower(regexp_replace(coalesce({name_col}, ''), "
        f"'[^a-zA-Z0-9 ]', '', 'g')))"
    )
    if kind == "name":
        return norm_name
    if kind == "token_sorted":
        return f"array_to_string(list_sort(str_split({norm_name}, ' ')), ' ')"
    if kind == "transliterated_name":
        return f"transliterated_business_name(coalesce({name_col}, ''))"
    if kind == "address":
        return (
            f"trim(lower(regexp_replace(coalesce({addr_col}, ''), "
            f"'[^a-zA-Z0-9 ]', '', 'g')))"
        )
    raise ValueError(f"Unknown SNM pass: {kind}")


def normalized_sql(column: str) -> str:
    """Use the name/address normalization implemented by the blocker."""
    return (
        f"trim(lower(regexp_replace(coalesce({column}, ''), "
        f"'[^a-zA-Z0-9 ]', '', 'g')))"
    )


def create_ranked_records(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    source_name: str,
    pass_name: str,
    window: int,
) -> str:
    """Create the same per-country rank and bucket values as build_candidates()."""
    ranked_table = f"ranked_{source_name}_{pass_name}"
    s1_sort_key = sort_key_sql(pass_name)
    source_sort_key = sort_key_sql(pass_name)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {ranked_table} AS
        SELECT entity_id, country, src,
               (ROW_NUMBER() OVER (PARTITION BY country ORDER BY sort_key, entity_id) - 1)
                   // {window} AS bucket
        FROM (
            SELECT entity_id, country, {s1_sort_key} AS sort_key, 'A' AS src
            FROM train_s1
            UNION ALL
            SELECT entity_id, country, {source_sort_key} AS sort_key, 'B' AS src
            FROM {source_table}
        );
    """)
    return ranked_table


def create_pass_status(
    con: duckdb.DuckDBPyConnection,
    source_name: str,
    ranked_table: str,
    pass_name: str,
) -> str:
    """Check only true pairs against one pass's exact bucket-join condition."""
    status_table = f"status_{source_name}_{pass_name}"
    source_prefix = source_name.upper() + "-"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {status_table} AS
        SELECT
            gt.s1_id,
            gt.source_entity_id,
            CASE
                WHEN s1.country = source.country
                 AND abs(s1.bucket - source.bucket) <= 1
                THEN TRUE
                ELSE FALSE
            END AS recovered
        FROM ground_truth_pairs gt
        LEFT JOIN {ranked_table} s1
            ON s1.entity_id = gt.s1_id AND s1.src = 'A'
        LEFT JOIN {ranked_table} source
            ON source.entity_id = gt.source_entity_id AND source.src = 'B'
        WHERE gt.source_entity_id LIKE '{source_prefix}%';
    """)
    return status_table


def create_source_diagnostics(
    con: duckdb.DuckDBPyConnection,
    source_name: str,
    source_table: str,
    status_tables: dict[str, str],
) -> str:
    """Combine pass outcomes and attach diagnostic fields for true pairs."""
    diagnostics_table = f"diagnostics_{source_name}"
    source_prefix = source_name.upper() + "-"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {diagnostics_table} AS
        SELECT
            gt.s1_id,
            gt.source_entity_id,
            coalesce(name_status.recovered, FALSE) AS recovered_by_name,
            coalesce(token_status.recovered, FALSE) AS recovered_by_token_sorted,
            coalesce(address_status.recovered, FALSE) AS recovered_by_address,
            coalesce(transliterated_status.recovered, FALSE) AS recovered_by_transliterated_name,
            coalesce(name_status.recovered, FALSE)
                OR coalesce(token_status.recovered, FALSE)
                OR coalesce(address_status.recovered, FALSE) AS recovered_by_any_pass,
            s1.country AS s1_country,
            source.country AS source_country,
            coalesce(s1.business_name, '') AS s1_business_name,
            coalesce(source.business_name, '') AS source_business_name,
            coalesce(s1.business_address, '') AS s1_address,
            coalesce(source.business_address, '') AS source_address,
            {normalized_sql('s1.business_name')} AS s1_normalized_name,
            {normalized_sql('source.business_name')} AS source_normalized_name,
            {normalized_sql('s1.business_address')} AS s1_normalized_address,
            {normalized_sql('source.business_address')} AS source_normalized_address
        FROM ground_truth_pairs gt
        LEFT JOIN train_s1 s1 ON s1.entity_id = gt.s1_id
        LEFT JOIN {source_table} source ON source.entity_id = gt.source_entity_id
        LEFT JOIN {status_tables['name']} name_status
            ON name_status.s1_id = gt.s1_id
           AND name_status.source_entity_id = gt.source_entity_id
        LEFT JOIN {status_tables['token_sorted']} token_status
            ON token_status.s1_id = gt.s1_id
           AND token_status.source_entity_id = gt.source_entity_id
        LEFT JOIN {status_tables['address']} address_status
            ON address_status.s1_id = gt.s1_id
           AND address_status.source_entity_id = gt.source_entity_id
        LEFT JOIN {status_tables['transliterated_name']} transliterated_status
            ON transliterated_status.s1_id = gt.s1_id
           AND transliterated_status.source_entity_id = gt.source_entity_id
        WHERE gt.source_entity_id LIKE '{source_prefix}%';
    """)
    return diagnostics_table


def print_recovery_summary(con: duckdb.DuckDBPyConnection, source_label: str, table: str) -> None:
    """Print exact per-pass and combined recall counts for one source."""
    (
        total,
        by_name,
        by_token,
        by_address,
        by_existing,
        by_transliterated_name,
        by_any_four,
        transliteration_only,
        transliteration_with_name,
        transliteration_with_token,
        transliteration_with_address,
        transliteration_with_existing,
        all_four,
        missed,
    ) = con.execute(f"""
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE recovered_by_name),
            COUNT(*) FILTER (WHERE recovered_by_token_sorted),
            COUNT(*) FILTER (WHERE recovered_by_address),
            COUNT(*) FILTER (WHERE recovered_by_any_pass),
            COUNT(*) FILTER (WHERE recovered_by_transliterated_name),
            COUNT(*) FILTER (
                WHERE recovered_by_any_pass OR recovered_by_transliterated_name
            ),
            COUNT(*) FILTER (
                WHERE recovered_by_transliterated_name AND NOT recovered_by_any_pass
            ),
            COUNT(*) FILTER (
                WHERE recovered_by_transliterated_name AND recovered_by_name
            ),
            COUNT(*) FILTER (
                WHERE recovered_by_transliterated_name AND recovered_by_token_sorted
            ),
            COUNT(*) FILTER (
                WHERE recovered_by_transliterated_name AND recovered_by_address
            ),
            COUNT(*) FILTER (
                WHERE recovered_by_transliterated_name AND recovered_by_any_pass
            ),
            COUNT(*) FILTER (
                WHERE recovered_by_transliterated_name
                  AND recovered_by_name
                  AND recovered_by_token_sorted
                  AND recovered_by_address
            ),
            COUNT(*) FILTER (
                WHERE NOT recovered_by_any_pass
                  AND NOT recovered_by_transliterated_name
            )
        FROM {table};
    """).fetchone()
    existing_recall = by_existing / total if total else 0.0
    all_four_recall = by_any_four / total if total else 0.0
    missed_by_existing = total - by_existing

    print(f"\n--- S1 -> {source_label} (PRE-CAP BLOCKING RECALL) ---")
    print(f"Total true pairs:                 {total:,}")
    print(f"Recovered by name:                {by_name:,}")
    print(f"Recovered by token_sorted:        {by_token:,}")
    print(f"Recovered by address:             {by_address:,}")
    print(f"Recovered by at least one existing pass: {by_existing:,} ({existing_recall:.4%})")
    print(f"Missed by all three existing passes: {missed_by_existing:,}")
    print(f"Recovered by transliterated_name: {by_transliterated_name:,}")
    print(f"Recovered by existing + transliterated_name: {by_any_four:,} ({all_four_recall:.4%})")
    print(f"Additional true pairs recovered by transliteration: {transliteration_only:,}")
    print(f"Percentage-point improvement:     {(all_four_recall - existing_recall) * 100:.4f}")
    print(f"Missed by all four passes:        {missed:,}")
    print("Transliteration overlap:")
    print(f"  transliteration only:           {transliteration_only:,}")
    print(f"  transliteration + name:         {transliteration_with_name:,}")
    print(f"  transliteration + token_sorted: {transliteration_with_token:,}")
    print(f"  transliteration + address:      {transliteration_with_address:,}")
    print(f"  transliteration + existing pass:{transliteration_with_existing:,}")
    print(f"  all four passes:                {all_four:,}")


def print_miss_diagnostics(con: duckdb.DuckDBPyConnection, source_label: str, table: str) -> None:
    """Summarize attributes of pairs that fail every blocking pass."""
    result = con.execute(f"""
        SELECT
            COUNT(*) AS missed_pairs,
            COUNT(*) FILTER (WHERE recovered_by_transliterated_name) AS rescued_by_transliteration,
            COUNT(*) FILTER (WHERE s1_address = '') AS s1_address_missing,
            COUNT(*) FILTER (WHERE source_address = '') AS source_address_missing,
            COUNT(*) FILTER (WHERE s1_address = '' OR source_address = '') AS either_address_missing,
            COUNT(*) FILTER (WHERE s1_country IS DISTINCT FROM source_country) AS country_mismatch,
            COUNT(*) FILTER (WHERE s1_normalized_name = source_normalized_name) AS exact_normalized_name,
            COUNT(*) FILTER (WHERE s1_normalized_address = source_normalized_address) AS exact_normalized_address,
            AVG(abs(length(s1_normalized_name) - length(source_normalized_name))) AS avg_name_length_difference,
            AVG(abs(length(s1_normalized_address) - length(source_normalized_address))) AS avg_address_length_difference
        FROM {table}
        WHERE NOT recovered_by_any_pass;
    """).fetchone()

    (
        missed,
        rescued_by_transliteration,
        s1_missing,
        source_missing,
        either_missing,
        country_mismatch,
        exact_name,
        exact_address,
        avg_name_length_difference,
        avg_address_length_difference,
    ) = result

    print(f"\nAll-three-missed diagnostics for S1 -> {source_label}")
    if not missed:
        print("  No all-three-missed pairs.")
        return

    def percentage(count: int) -> str:
        return f"{count:,} ({count / missed:.2%})"

    print(f"  Missed pairs:                         {missed:,}")
    print(f"  Recovered by transliterated_name:     {rescued_by_transliteration:,}")
    print(f"  S1 address missing:                   {percentage(s1_missing)}")
    print(f"  Source address missing:               {percentage(source_missing)}")
    print(f"  Either address missing:               {percentage(either_missing)}")
    print(f"  S1/source country mismatch:           {percentage(country_mismatch)}")
    print(f"  Exactly equal normalized names:       {percentage(exact_name)}")
    print(f"  Exactly equal normalized addresses:   {percentage(exact_address)}")
    print(f"  Average normalized-name length diff:  {avg_name_length_difference:.2f}")
    print(f"  Average normalized-address length diff:{avg_address_length_difference:.2f}")


def display_value(value: str) -> str:
    """Keep tabular sample output on one line per record."""
    return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def print_transliteration_rescue_sample(
    con: duckdb.DuckDBPyConnection, source_label: str, table: str
) -> None:
    """Print original-three-pass misses that the additive pass recovers."""
    rows = con.execute(f"""
        SELECT
            s1_id,
            source_entity_id,
            s1_business_name,
            source_business_name,
            s1_address,
            source_address,
            s1_country,
            source_country,
            recovered_by_name,
            recovered_by_token_sorted,
            recovered_by_address,
            recovered_by_transliterated_name
        FROM {table}
        WHERE NOT recovered_by_any_pass
          AND recovered_by_transliterated_name
        ORDER BY s1_id, source_entity_id
        LIMIT {SAMPLE_SIZE};
    """).fetchall()

    print(
        f"\nSample of up to {SAMPLE_SIZE} all-three-missed but transliteration-recovered "
        f"S1 -> {source_label} pairs"
    )
    if not rows:
        print("  No rows to display.")
        return

    print(
        "s1_id\tsource_id\ts1_business_name\tsource_business_name\t"
        "transliterated_s1_name\ttransliterated_source_name\t"
        "s1_country\tsource_country\tname_pass\ttoken_sorted_pass\t"
        "address_pass\ttransliterated_name_pass"
    )
    for row in rows:
        (
            s1_id,
            source_id,
            s1_name,
            source_name,
            s1_address,
            source_address,
            s1_country,
            source_country,
            name_pass,
            token_sorted_pass,
            address_pass,
            transliterated_pass,
        ) = row
        fields = (
            s1_id,
            source_id,
            s1_name,
            source_name,
            transliterated_name(s1_name),
            transliterated_name(source_name),
            s1_country or "",
            source_country or "",
            name_pass,
            token_sorted_pass,
            address_pass,
            transliterated_pass,
        )
        print("\t".join(display_value(str(value)) for value in fields))


def main() -> None:
    train_s1_path = locate_training_file("train_source1.tsv")
    train_s2_path = locate_training_file("train_source2.tsv")
    train_s3_path = locate_training_file("train_source3.tsv")
    ground_truth_path = locate_training_file("train_ground_truth.tsv")

    print("--- ANALYZING fast_inference.py PRE-CAP BLOCKING MISSES ---")
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DB_MEMORY_LIMIT}';")
    con.execute(f"SET threads = {DB_THREADS};")
    con.execute("SET preserve_insertion_order = false;")
    register_transliteration_udf(con)

    con.execute(f"""
        CREATE OR REPLACE TABLE train_s1 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(train_s1_path)}', delim='\t', header=True, auto_detect=True);

        CREATE OR REPLACE TABLE train_s2 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(train_s2_path)}', delim='\t', header=True, auto_detect=True);

        CREATE OR REPLACE TABLE train_s3 AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{sql_path(train_s3_path)}', delim='\t', header=True, auto_detect=True);

        CREATE OR REPLACE TABLE train_ground_truth AS
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{sql_path(ground_truth_path)}', delim='\t', header=True, auto_detect=True);

        CREATE OR REPLACE TABLE ground_truth_pairs AS
        SELECT
            source1_entity_id AS s1_id,
            trim(matched_entity_id) AS source_entity_id
        FROM train_ground_truth,
             UNNEST(str_split(coalesce(matched_entity_ids, ''), ',')) AS matches(matched_entity_id)
        WHERE trim(matched_entity_id) <> '';
    """)

    for source_name, source_table in (("s2", "train_s2"), ("s3", "train_s3")):
        status_tables: dict[str, str] = {}
        for pass_name, config in ALL_PASSES.items():
            ranked_table = create_ranked_records(
                con, source_table, source_name, pass_name, config["window"]
            )
            status_tables[pass_name] = create_pass_status(
                con, source_name, ranked_table, pass_name
            )

        diagnostics_table = create_source_diagnostics(
            con, source_name, source_table, status_tables
        )
        source_label = source_name.upper()
        print_recovery_summary(con, source_label, diagnostics_table)
        print_miss_diagnostics(con, source_label, diagnostics_table)
        print_transliteration_rescue_sample(con, source_label, diagnostics_table)


if __name__ == "__main__":
    main()
