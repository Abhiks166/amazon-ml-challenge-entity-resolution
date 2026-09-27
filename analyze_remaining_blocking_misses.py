"""Diagnose signals in true pairs missed by all four experimental blockers.

This is a ground-truth-oriented experiment: it reuses the four-pass audit in
analyze_blocking_misses.py to identify misses, then streams only those true
pairs. It never materializes an S1 x S2/S3 candidate-pair join.

Run with:
    python analyze_remaining_blocking_misses.py
"""

from collections import Counter
import random
import re

import duckdb
from rapidfuzz import fuzz

from analyze_blocking_misses import (
    ALL_PASSES,
    DB_MEMORY_LIMIT,
    DB_THREADS,
    create_pass_status,
    create_ranked_records,
    create_source_diagnostics,
    locate_training_file,
    register_transliteration_udf,
    sql_path,
)


SAMPLE_SIZE = 50
FETCH_SIZE = 10_000
RANDOM_SEED = 2026

EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s,;]+|\b[a-z0-9][a-z0-9-]*\.[a-z]{2,}(?:/[^\s,;]*)?",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d() .-]{5,}\d)(?!\d)")
NUMBER_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def make_ngrams(value: str, size: int) -> set[str]:
    """Return unpadded character n-grams from an already normalized name."""
    if len(value) < size:
        return set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def compact_text(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def identifiers(value: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """Extract lightweight structured strings found within name/address text."""
    lowered = value.lower()
    emails = set(EMAIL_RE.findall(lowered))
    domains_or_urls = set(URL_RE.findall(lowered))
    phones = {
        re.sub(r"\D", "", match)
        for match in PHONE_RE.findall(lowered)
        if len(re.sub(r"\D", "", match)) >= 7
    }
    numbers = set(NUMBER_RE.findall(lowered))
    return emails, domains_or_urls, phones, numbers


def reservoir_add(sample: list[tuple], row: tuple, seen: int, rng: random.Random) -> None:
    """Maintain a deterministic uniform sample without keeping all misses."""
    if len(sample) < SAMPLE_SIZE:
        sample.append(row)
        return
    replacement = rng.randrange(seen)
    if replacement < SAMPLE_SIZE:
        sample[replacement] = row


def create_ground_truth_pairs(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE ground_truth_pairs AS
        SELECT
            source1_entity_id AS s1_id,
            trim(matched_entity_id) AS source_entity_id
        FROM train_ground_truth,
             UNNEST(str_split(coalesce(matched_entity_ids, ''), ',')) AS matches(matched_entity_id)
        WHERE trim(matched_entity_id) <> '';
    """)


def create_remaining_miss_table(
    con: duckdb.DuckDBPyConnection, source_name: str, source_table: str
) -> str:
    """Find true pairs missed by all current passes without candidate joins."""
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
    misses_table = f"remaining_misses_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {misses_table} AS
        SELECT *
        FROM {diagnostics_table}
        WHERE NOT recovered_by_any_pass
          AND NOT recovered_by_transliterated_name;
    """)
    return misses_table


def analyze_source(con: duckdb.DuckDBPyConnection, source_name: str, misses_table: str) -> None:
    """Stream all-four misses and summarize potential additive signals."""
    total = con.execute(f"SELECT COUNT(*) FROM {misses_table}").fetchone()[0]
    print(f"\n--- S1 -> {source_name.upper()} remaining PRE-CAP misses: {total:,} ---")
    if not total:
        return

    counters = Counter()
    sums = Counter()
    ngram_masks = Counter()
    sample: list[tuple] = []
    rng = random.Random(RANDOM_SEED)

    cursor = con.execute(f"""
        SELECT
            s1_id, source_entity_id,
            s1_business_name, source_business_name,
            s1_address, source_address,
            s1_country, source_country,
            s1_normalized_name, source_normalized_name,
            s1_normalized_address, source_normalized_address
        FROM {misses_table};
    """)

    seen = 0
    while rows := cursor.fetchmany(FETCH_SIZE):
        for row in rows:
            (
                s1_id,
                source_id,
                s1_name,
                source_name_value,
                s1_address,
                source_address,
                s1_country,
                source_country,
                s1_norm_name,
                source_norm_name,
                s1_norm_address,
                source_norm_address,
            ) = row
            seen += 1

            same_country = s1_country == source_country and s1_country is not None
            grams = {
                size: bool(make_ngrams(s1_norm_name, size) & make_ngrams(source_norm_name, size))
                for size in (3, 4, 5)
            }
            for size, recovered in grams.items():
                if same_country and recovered:
                    counters[f"char_{size}"] += 1
            mask = "".join(str(size) for size in (3, 4, 5) if same_country and grams[size]) or "none"
            ngram_masks[mask] += 1

            name_ratio = fuzz.ratio(s1_norm_name, source_norm_name)
            name_token_set = fuzz.token_set_ratio(s1_norm_name, source_norm_name)
            address_ratio = fuzz.ratio(s1_norm_address, source_norm_address)
            sums["name_ratio"] += name_ratio
            sums["name_token_set"] += name_token_set
            sums["address_ratio"] += address_ratio

            s1_tokens = set(TOKEN_RE.findall(s1_norm_name))
            source_tokens = set(TOKEN_RE.findall(source_norm_name))
            shared_tokens = s1_tokens & source_tokens
            if shared_tokens:
                counters["shared_name_token"] += 1
            if s1_tokens or source_tokens:
                sums["token_jaccard"] += len(shared_tokens) / len(s1_tokens | source_tokens)

            s1_email, s1_url, s1_phone, s1_numbers = identifiers(f"{s1_name} {s1_address}")
            source_email, source_url, source_phone, source_numbers = identifiers(
                f"{source_name_value} {source_address}"
            )
            if s1_email & source_email:
                counters["shared_email"] += 1
            if s1_url & source_url:
                counters["shared_url"] += 1
            if s1_phone & source_phone:
                counters["shared_phone"] += 1
            if s1_numbers & source_numbers:
                counters["shared_number"] += 1
            if not same_country:
                counters["country_mismatch"] += 1
            if s1_norm_name == source_norm_name:
                counters["exact_normalized_name"] += 1
            if s1_norm_address == source_norm_address:
                counters["exact_normalized_address"] += 1

            reservoir_add(
                sample,
                (
                    s1_id,
                    source_id,
                    s1_name,
                    source_name_value,
                    s1_address,
                    source_address,
                    s1_country or "",
                    source_country or "",
                    name_ratio,
                    address_ratio,
                    mask,
                    ",".join(sorted(shared_tokens)) or "-",
                    ",".join(sorted(s1_numbers & source_numbers)) or "-",
                ),
                seen,
                rng,
            )

    combined = total - ngram_masks["none"]
    print("Character n-gram recovery if used as an additional same-country name blocker:")
    for size in (3, 4, 5):
        value = counters[f"char_{size}"]
        print(f"  char-{size}: {value:,} ({value / total:.4%})")
    print(f"  combined char-3/4/5: {combined:,} ({combined / total:.4%})")
    print("  exact char-n overlap masks (within the same country):")
    for mask in ("345", "34", "35", "45", "3", "4", "5", "none"):
        value = ngram_masks[mask]
        print(f"    {mask}: {value:,} ({value / total:.4%})")

    print("Useful signals among the remaining misses:")
    print(f"  average normalized-name ratio: {sums['name_ratio'] / total:.2f}")
    print(f"  average normalized-name token-set ratio: {sums['name_token_set'] / total:.2f}")
    print(f"  average normalized-address ratio: {sums['address_ratio'] / total:.2f}")
    print(f"  average shared-name-token Jaccard: {sums['token_jaccard'] / total:.4f}")
    for label, key in (
        ("pairs with at least one shared name token", "shared_name_token"),
        ("pairs with shared address/name number", "shared_number"),
        ("pairs with an exact shared email", "shared_email"),
        ("pairs with an exact shared URL/domain", "shared_url"),
        ("pairs with an exact shared phone", "shared_phone"),
        ("country mismatches", "country_mismatch"),
        ("exactly equal normalized names", "exact_normalized_name"),
        ("exactly equal normalized addresses", "exact_normalized_address"),
    ):
        value = counters[key]
        print(f"  {label}: {value:,} ({value / total:.4%})")

    print(f"\nRepresentative sample of up to {SAMPLE_SIZE} remaining misses")
    print(
        "s1_id\tsource_id\ts1_business_name\tsource_business_name\t"
        "s1_address\tsource_address\ts1_country\tsource_country\t"
        "name_ratio\taddress_ratio\tchar_ngrams\tshared_name_tokens\tshared_numbers"
    )
    for row in sorted(sample, key=lambda item: (item[0], item[1])):
        output = list(row[:8]) + [f"{row[8]:.1f}", f"{row[9]:.1f}"] + list(row[10:])
        print("\t".join(compact_text(str(value)) for value in output))


def main() -> None:
    train_s1_path = locate_training_file("train_source1.tsv")
    train_s2_path = locate_training_file("train_source2.tsv")
    train_s3_path = locate_training_file("train_source3.tsv")
    ground_truth_path = locate_training_file("train_ground_truth.tsv")

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
    """)
    create_ground_truth_pairs(con)

    print("--- REMAINING PRE-CAP BLOCKING-MISS SIGNAL ANALYSIS ---")
    for source_name, source_table in (("s2", "train_s2"), ("s3", "train_s3")):
        misses_table = create_remaining_miss_table(con, source_name, source_table)
        analyze_source(con, source_name, misses_table)


if __name__ == "__main__":
    main()
