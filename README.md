# Scalable Multi-Catalog Entity Resolution Pipeline

A high-performance, out-of-core Entity Resolution (Record Linkage) system designed for the Amazon ML Challenge. It efficiently processes ~17.2 trillion potential record pairs across multi-source business entity catalogs ($S1$, $S2$, and $S3$) on a constrained 16GB RAM CPU machine.

## 🚀 Key Highlights & Benchmarks
- **Search Space:** ~1.73M $S1$ anchor entities evaluated against ~4.88M $S2$ and ~5.08M $S3$ catalog records.
- **Candidate Blocking:** Bucketed Multi-Pass Sorted Neighborhood Method (SNM) in DuckDB (Raw Name, Token-Sorted Name, Address) generated deduped candidate pairs in ~20 seconds.
- **Inference Latency:** Streamed ~28,000 Parquet chunk files across a 4-worker multiprocessing pool in under 30 minutes.
- **Memory Footprint:** Enforced a hard ~4GB RAM limit via disk-backed DuckDB temporary tables (`PRAGMA temp_directory`).
- **Match Coverage:** Successfully resolved **1,731,959 out of 1,732,544 $S1$ entities (~99.97% coverage)**.

## 🛠️ Tech Stack
- **Database / SQL Engine:** DuckDB
- **Data Manipulation:** Polars
- **Machine Learning Matcher:** LightGBM
- **String Distance Metrics:** C++-accelerated `rapidfuzz`
- **Concurrency & Parallelism:** Python `concurrent.futures.ProcessPoolExecutor`
- **Storage Format:** Apache Parquet

## 📁 Repository Structure
- `train_model.py`: Trains LightGBM matcher using 10 fuzzy string-distance features.
- `fast_inference.py` / `scalable_inference.py`: Runs multi-pass SNM candidate blocking and streams chunk files to parallel workers.
- `export_submission.py`: Assembles scored Parquet result chunks from disk, applies probability capping (Top 6 per entity), and exports `submission.tsv`.
- `features.py` / `utils.py`: High-performance feature extraction routines.
