# Business Entity Resolution — ML Challenge 2026

Resolve which records across three noisy, multilingual business-record sources refer to the same
real-world business. Source 1 is the deduplicated reference; for every S1 entity we find its
matching S2 and S3 records.

Scored on **macro F_0.5** — precision-weighted 2× over recall, averaged over *all* S1 entities
including singletons.

See [`plan.md`](plan.md) for the measured dataset analysis, the architecture, and the rationale
behind each decision.

## Approach at a glance

```
normalize ─► embed (24.2M recs, 512d fp16, resumable) ─► FAISS, sharded by country
                                                              │
  lexical channels: digit-tokens · address 3-grams · name 3-grams
  dense channel:    ANN top-25 (cross-script / trade-name rescue)
                                                              │
                        union, cap K=50 ──► candidate_pairs.tsv
                                                              │
                     pair features ──► 4 models ──► compare
                                                              │
        threshold tuned for macro F_0.5 ──► exclusivity resolution
                                                              │
                                                  matching_results.tsv
```

Three measured facts drive the design:

- **Address is the dominant signal, not name.** Address token overlap is present in 99.98% of true
  matches; the name is absent or replaced in 12.2% (transliteration, or an unrelated trade name).
  So lexical address blocking carries recall and dense retrieval rescues the cross-script cases.
- **`country` never disagrees across a true match** (0 / 13,836 sampled pairs), so it is a hard
  partition — kept as an open set of labels, so the test-only `France` passes through unhardcoded.
- **Each S2/S3 record matches at most one S1 entity** (verified exactly over all 7,638,365 ground-truth
  pairs). Enforcing that at decision time removes false merges, which F_0.5 punishes double.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --extra dev
```

On macOS, LightGBM needs OpenMP: `brew install libomp`.

> **Why training runs one model per subprocess.** This venv ends up with four OpenMP
> runtimes (torch, scikit-learn, faiss, and LightGBM's own), and LightGBM segfaults
> outright — exit 139, inside `Dataset.__init_from_np2d` — if torch was imported first.
> `KMP_DUPLICATE_LIB_OK=TRUE` does not help. `ber train` therefore spawns one process per
> model and resolves model classes lazily, so torch and LightGBM never share a process.
> If you add a model, keep it out of any eagerly-imported module.

The dataset is not in the repo. Place it at `student_resource/dataset/{train,test}/`.

### Embedding server

The embedder talks to an OpenAI-compatible `/v1/embeddings` endpoint, so the same client code runs
against llama.cpp Metal locally and vLLM on CUDA in the cloud.

```bash
# Local (Apple Silicon, Metal-accelerated)
llama-server --hf-repo Qwen/Qwen3-Embedding-0.6B-GGUF \
             --hf-file Qwen3-Embedding-0.6B-Q8_0.gguf \
             --embedding --pooling last -ngl 99 --port 8080

# Remote / AWS (CUDA)
vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8080
```

Then point the pipeline at it via `embedding.backend` and `embedding.endpoint` in
`configs/default.yaml`.

> vLLM is **not** a local option on macOS: it forces `VLLM_TARGET_DEVICE=cpu` and benchmarks
> 20–30× slower than llama.cpp's Metal backend. At 24.2M records that is days versus hours.

## Reproduce end-to-end

```bash
# 0. Benchmark the embedder and project the full-run wall clock. Gates step 2.
uv run ber bench-embedder --sample 50000

# 1. Embed all 24.2M records (resumable; safe to interrupt and re-run)
uv run ber embed --split train
uv run ber embed --split test

# 2. Build the two vector collections, sharded by country
uv run ber index --split train     # base_vectors (S1) + comparable_vectors (S2+S3)
uv run ber index --split test

# 3. Candidate generation, and report the blocking recall ceiling
uv run ber block --split train --report-recall
uv run ber block --split test

# 4. Features + train all four models, each saved separately with stats JSON
uv run ber train --models lightgbm xgboost mlp siamese

# 5. Compare models on macro F_0.5 / precision / recall / AUC-ROC
uv run ber compare-models

# 6. Tune the decision threshold and pick the winning model
uv run ber tune --model best

# 7. Inference -> output/matching_results.tsv + output/candidate_pairs.tsv
uv run ber predict --split test

# 8. Validate with the official checker
cd student_resource && python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

Every knob lives in [`configs/default.yaml`](configs/default.yaml). The embedding instruction is
plain text in [`configs/instruction.txt`](configs/instruction.txt) — edit it freely; nothing in the
code hardcodes it.

For a fast end-to-end smoke run on a fraction of the data:

```bash
uv run ber train --set training.sample_entities=20000   # ~1M pairs instead of ~10M
```

## Layout

```
configs/         default.yaml (all knobs) + instruction.txt (editable)
src/ber/
  config.py      YAML loading with resolved paths
  io.py          streaming TSV readers/writers, atomic writes
  normalize.py   scripts, accents, legal suffixes, address abbreviations, digit tokens
  eval.py        macro F_0.5 exactly as the PS defines it
  device.py      MPS -> CUDA -> CPU auto-select for the classifiers
  embed/         instruction loading, async /v1/embeddings client, backends, shard writer
  vectors/       VectorStore protocol; FAISS (default) and Chroma backends
  blocking/      lexical channels, dense channel, candidate union
  features.py    pair feature extraction (vectorized)
  loss.py        soft-F_beta surrogate + weighted BCE
  models/        lightgbm, xgboost, mlp, siamese, compare
  decide.py      threshold tuning + 1:1 exclusivity resolution
  cli.py         the `ber` entry point
scripts/         bench_embedder.py, make_submission.py
tests/           normalization, metric, decision-layer unit tests
```

## Tests

```bash
uv run pytest
```

79 tests covering the normalization rules case by case (accents, per-country abbreviations,
transliterated state names, digit folding, cross-script skeletons), the metric against the
problem statement's worked example (P=2/3, R=1.0 → 0.714), the soft-F_0.5 loss gradients,
channel fusion, and the exclusivity resolver.

### Fast iteration

The full training set is 2.2M entities against 10.3M comparable records, so build a subset
first — it keeps every true match, so recall measured on it is comparable:

```bash
uv run python scripts/make_subset.py --entities 20000 --distractors 400000
uv run ber train --set paths.data_root=student_resource/dataset_subset \
                 --set blocking.dense_ann.enabled=false
```

Measured on that subset with the dense channel off: blocking pair recall **99.21%** at K=50,
best model held-out macro F_0.5 **0.9795** (precision 0.9921 / recall 0.9641). See
[`plan.md` §6](plan.md) for the full table and the known `entity_attn` regression.

## Fair play

No external data, APIs, or geocoding services are used at any point. Transliteration is done
offline with `unidecode`. All models are Apache-2.0 or MIT and under 8B parameters
(`Qwen/Qwen3-Embedding-0.6B` is 595M, Apache-2.0).
