# Business Entity Resolution — Architecture & Implementation Plan

> Reference document for the ML Challenge 2026 Business Entity Resolution solution.
> Records the measured dataset facts, the decisions taken, and the rationale — so the
> *why* survives past the conversation that produced it.

---

## 1. Context

`CLAUDE.md` specifies a pipeline: embed business records with an instruction-aware multilingual
model, store them in a vector DB, retrieve ~50 nearest candidates per Source-1 entity, and classify
those candidate pairs with 4 models (2 NN + 2 tree) to produce `matching_results.tsv` and
`candidate_pairs.tsv`.

The dataset was measured before implementation. Several findings changed the original design, and
they are recorded here because they are expensive to re-derive.

### 1.1 Measured dataset facts

| Fact | Value | Measurement |
|---|---|---|
| Train records | S1 **2,206,821** · S2 **5,034,616** · S3 **5,285,603** | `wc -l` minus header; verified 4 fields/row, no embedded newlines, no duplicate IDs |
| Test records | S1 **1,732,544** · S2 **4,887,273** · S3 **5,082,316** | same |
| **Total to embed** | **24,229,173** | train + test, all three sources |
| Ground-truth pairs | 7,638,365 across 2,206,821 S1 entities | mean **3.46** matches/entity, max **11** |
| Singletons | **5.58%** (123,247 entities) | worth a full 1.0 each under macro F_0.5 |
| Per-entity ceiling | ≤ **5** S2 and ≤ **6** S3 matches | GT distribution |
| `country` agreement | **0 disagreements / 13,836** sampled true pairs | safe hard partition |
| **Exclusivity** | each S2/S3 record matches **at most one** S1 | 7,638,365 IDs, 7,638,365 mentions, every count == 1 |
| Address signal | addr token Jaccard == 0 in only **0.02%** of true matches | address is the dominant field |
| Name signal | name token Jaccard == 0 in **12.2%** of true matches | cross-script or DBA/trade-name replacement |
| Digit tokens | **80.3%** of true matches share a digit token | house number / PIN — high-selectivity key |
| Lexical recall ceiling | ≈ **99.4%** | only 0.58% of matches have name *and* addr Jaccard ≤ 0.2 |
| Empty addresses | **3.36%** of S2, **3.33%** of S3, 0% of S1 | name-only matching required for these |
| Test-only country | **France**, 259,452 S1 entities (15%) | unseen in training |
| Test country mix | India 809,986 · US 663,106 · France 259,452 (S1) | |

### 1.2 Hardware

M3 Max (`Mac15,10`), 14 cores (10P/4E), **36 GB unified memory**, ~640 GB free disk.
**No CUDA** — Metal/MPS only.

### 1.3 Representative noise, from the data

```
S1  Raj Investments LLP           | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu
S2  ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி  | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu
S3  Raj Investments எல்எல்பி       | 6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN
S3  ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி  | 6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு
```
Name script varies (Latin / Tamil / mixed); **the address stays Latin** and matches almost exactly.
The state is sometimes transliterated (`தமிழ்நாடு`) or abbreviated (`TN`).

```
S1  Maure Williams Colombier Inc  | 85 Wayne Avenue, Ticonderoga, NY
S3  Dréxkor                       | 85 Wanye Avenue, Ticonderoga Townshiip, New York
```
A **completely unrelated trade name**, matched only by address — with a typo (`Wayne`→`Wanye`)
and an expanded state (`NY`→`New York`). Name-only blocking cannot reach this pair.

### 1.4 What these facts imply

1. **A 1.5B-param embedder at 1536d is not viable here.** 24.2M × 1536 × 2B = **74 GB** of fp16
   vectors against 36 GB of RAM, and days of MPS compute.
2. **vLLM cannot accelerate this on macOS.** It forces `VLLM_TARGET_DEVICE=cpu`, requires a source
   build, and benchmarks at 3–5 tok/s versus llama.cpp Metal's 92 — a 20–30× gap.
3. **Embedding is ~99% of the project's compute**, not the classifier. The classifier trains on
   ~40-dim feature vectors (LightGBM: CPU, minutes; small MLPs: minutes on MPS). The accelerator
   belongs to the embedder.
4. **Dense-only retrieval aims at the weaker field.** Address token overlap alone reaches a ~99.4%
   recall ceiling; names are absent or replaced in 12.2% of matches. Dense retrieval earns its keep
   specifically on cross-script names, where no string metric works — so it is a *channel*, not the
   whole blocker.
5. **The exclusivity constraint is free precision.** A per-pair classifier cannot see that two S1
   entities are competing for one S2 record; enforcing it at decision time removes false merges,
   which F_0.5 punishes at 2× the weight of misses.

> **Gotcha:** opening these TSVs in Numbers or Excel silently truncates them — Numbers caps at
> exactly 1,000,000 rows, Excel at 1,048,576. `train_source1.tsv` really has 2,206,821 records.

---

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Embedder | **`Qwen/Qwen3-Embedding-0.6B`** | Apache-2.0, instruction-aware, 100+ languages, 32k ctx, and uniquely supports **MRL** dimension truncation |
| Dimension | **512d** (MRL-truncated from 1024) | 24.2M × 512 × 2B = **24.8 GB** fp16; halves storage at small quality cost |
| Runtime | **llama.cpp Metal** server on `/v1/embeddings`; **vLLM adapter** behind the same interface | Metal GPU locally, and the identical client code targets vLLM on AWS CUDA later |
| Classifier device | MPS → CUDA → CPU auto-select | per `CLAUDE.md` |
| Blocking | **Lexical-first hybrid** + dense ANN union | ~99.4% ceiling from cheap channels, dense rescues cross-script |
| Vector store | **FAISS** default, sharded by country; **ChromaDB** behind the same `VectorStore` API | FAISS handles 4.7M-vector shards on 36 GB; Chroma's SQLite metadata layer does not |
| Instruction | editable **`configs/instruction.txt`** | user revises it without touching code |
| Training pairs | **~10M** (≈200k sampled S1 entities × ~50); 1M fallback config | learn nuance without 110M pairs |
| Sequencing | **full 24.2M embed before any scoring** | user's call |

**Models considered and rejected:** `jina-embeddings-v3` (CC-BY-NC-4.0 violates the MIT/Apache-2.0
rule), `bge-m3` (no instruction support), `multilingual-e5-small` / `gte-multilingual-base` (fixed
`query:`/`passage:` prefixes, not instructions).

---

## 3. Pipeline

```
 raw TSVs ──► normalize ──► embed (24.2M, 512d fp16, resumable)
                               │
                               ├──► base_vectors        (S1)
                               └──► comparable_vectors  (S2+S3)   sharded by country
                                            │
 one hashed inverted index, four weighted key types:
   d digit tokens (house no./PIN)   w=1.3     │
   a address tokens                 w=1.2     ├──► RRF fuse, cap K=50 ──► candidate_pairs.tsv
   n name tokens                    w=1.0     │
   s romanized squeezed name        w=0.9     │
   g name char-3grams               w=0.6     │
   + dense ANN top-25 (cross-script rescue) ──┘
                                            │
                                    pair features (49)
                                            │
                        ┌───────────────────┼───────────────────┐
                   LightGBM            XGBoost        MLP / entity_attn
                    (one subprocess each — OpenMP isolation, see §7)
                        └───────────────────┼───────────────────┘
                                   compare.py → stats JSON
                                            │
                              threshold τ (macro F_0.5)
                                            │
                              exclusivity resolution (1:1)
                                            │
                                  matching_results.tsv
```

### Stage 0 — Scaffold + embedder benchmark (gate)

`uv` project, `src/ber/` layout, `configs/default.yaml` holding every knob.
`configs/instruction.txt` holds the base embedding instruction as editable plain text.

`scripts/bench_embedder.py` encodes a 50k sample and reports records/sec, the **projected
wall-clock for all 24.2M**, and cross-script recall@50 at 512d. **This gates Stage 2** — no
10-hour job is launched on an estimate.

`src/ber/eval.py` (macro F_0.5, exactly as the PS defines it) is written first so every later
stage is measurable.

### Stage 1 — Normalization

`src/ber/normalize.py`, shared by blocking, features, and embedding-text construction:

- NFKC, accent folding (France: `Àmicale`→`amicale`, `Hêtres`→`hetres`), case, punctuation.
- Legal suffixes as **normalized variants, never deletions**: `Pvt/Private`, `Ltd/Limited`,
  `Corp/Corporation`, `LLP`, `LLC`, `SARL/SASU/SCI/SAS/S.A.`
- Address abbreviations: `Rd/Road`, `St/Street`, `AV/Avenue`, `R./Rue`; US state ↔ 2-letter;
  Indian state ↔ abbreviation **including transliterated forms** (`उत्तर प्रदेश`, `தமிழ்நாடு`).
- Script detection + **offline** transliteration of Devanagari/Tamil names to Latin.
  Local libraries only — the fair-play rule forbids external lookups.
- Digit-token extraction with leading-zero folding, so `AF-684` ≡ `AF-0684`.

### Stage 2 — Full embedding pass

`src/ber/embed/` — async batched `/v1/embeddings` client, pluggable backend
(`llamacpp` | `vllm` | `torch-mps`), shard writer.

**Resumable and idempotent**: fixed-size shards keyed by `(split, source, country)`, a JSON
manifest of completed shards, atomic writes. A multi-hour laptop run *will* be interrupted; a
restart must resume rather than recompute. Output: fp16 `.npy` memmaps at 512d + aligned `ids.npy`.

### Stage 3 — Vector store

`VectorStore` protocol (`add`, `search`) with `FaissStore` (default) and `ChromaStore` (same API).
Two collections — **`base_vectors`** (S1) and **`comparable_vectors`** (S2+S3) — each sharded by
country so a search never crosses a partition. At 512d fp16: India 4.72M → 4.8 GB,
US 3.82M → 3.9 GB, France 1.43M → 1.5 GB. Persisted to disk; embeddings computed once.

### Stage 4 — Candidate generation

Per S1 entity, **within its country**:

One hashed inverted index carries all key types, each typed by prefix so an address token can
never match a name token, and each weighted (weights tuned by sweep — see §6):

| Key | Weight | Rationale |
|---|---|---|
| `d:` digit tokens | 1.3 | 80.3% of true matches share one |
| `a:` address tokens | 1.2 | the dominant field |
| `n:` name tokens | 1.0 | same-script name matches |
| `s:` romanized, repeat-squeezed name | 0.9 | **cross-script bridge** |
| `g:` name character 3-grams | 0.6 | domain-style names, single-char typos |
| dense ANN top-25 | — | **cross-script + trade-name rescue** |

Character n-grams are used on **names only**. A char-3gram TF-IDF cosine over 4.7M *addresses*
does not scale: common trigrams like `" st"` link a query to nearly every record, so the
sparse product has billions of nonzeros. Word tokens with aggressive document-frequency
filtering keep it genuinely sparse. Trigrams over addresses instead appear as *features* on the
≤50 surviving candidates, where they cost nothing.

Channels are fused with **Reciprocal Rank Fusion**, not by adding scores: IDF-weighted hashed
cosines and embedding cosines are not on a comparable scale, and RRF needs only the ordering
within each channel. Reports **blocking recall ceiling** and **reduction ratio** against GT —
the metric that bounds everything downstream.

### Stage 5 — Pair features

`src/ber/features.py`, vectorized and parallel across 14 cores.

- **Name:** token Jaccard, **IDF-weighted token overlap**, char-3gram cosine, rapidfuzz
  ratio/partial/token_set, acronym & initials match, length ratio, suffix-stripped variants.
- **Address:** token Jaccard, IDF-weighted overlap, **digit-token set intersection**, char-3gram
  cosine, city/state agreement, `has_address` flag.
- **Dense:** cosine at 512d, ANN rank, ANN score.
- **Transliteration:** script-mismatch flag, romanized-name similarity.
- **Structural:** candidate rank, block size, score margin vs. the entity's best sibling, and
  **contention count** — how many other S1 entities claim this record.

> The `Helio Tech Pvt` vs `Helio Constructions Pvt` case is resolved by the **IDF-weighted
> name-token** feature: `Helio` is shared but the rare tokens `Tech` / `Constructions` mismatch.
> The embedding instruction shapes *retrieval*; this feature makes the *decision*. Tuning only the
> instruction will move the score very little.

### Stage 6 — Models

One file per model under `src/ber/models/`, each trained and **saved individually** with stats
to JSON.

- **Trees:** LightGBM, XGBoost (`scale_pos_weight < 1`, biasing toward precision).
- **NNs:** MLP over the feature vector; siamese/cross-encoder head over pair embeddings + features.
  Device auto-select MPS → CUDA → CPU.
- **Loss derived from the eval criterion** (`src/ber/loss.py`): batches are grouped **by S1 entity**
  so a differentiable **soft-F_0.5 surrogate** can be computed per entity from sigmoid
  probabilities and optimised as `1 − soft_F_0.5`. Weighted BCE with an FP penalty is the
  baseline/ablation.
- Training set ~10M pairs from ~200k randomly sampled S1 entities, all positives + hard negatives.
  **Split by entity, never by pair**, to avoid leakage.
- `compare.py` — comparator script emitting **macro F_0.5** (primary) plus precision, recall,
  AUC-ROC into one stats JSON + comparison table.

### Stage 7 — Decision layer

Largest F_0.5 lever after blocking (`src/ber/decide.py`):

1. **Threshold τ** tuned to maximise macro F_0.5 on the held-out entity split — *not* AUC, since
   F_0.5 is what is scored and is not differentiable.
2. **Exclusivity resolution** — each S2/S3 record belongs to at most one S1 entity (verified
   exactly). Contested records are resolved by score over connected components.
3. **Singletons** — emit empty when nothing clears τ; 5.58% of entities, each worth a full 1.0.
4. Optional per-entity cap informed by the GT ceiling (≤5 S2, ≤6 S3).

### Stage 8 — Inference & deliverables

- Test inference: 1.73M S1 × ~50 ≈ **87M pairs** → both TSVs in `output/`.
- `student_resource/utils/validate_submission.py` must print `PASS` (reused as-is, with `--check-ids`).
- `Documentation_template.md` filled: methodology, blocking strategy with recall ceiling and
  reduction ratio, features, architecture, threshold selection, error analysis.
- `scripts/make_submission.py` → required zip layout.
- `docker-compose.yml` with the two services from `CLAUDE.md`: **embedder** and **trainer**.

---

## 4. Verification

1. `uv run ber bench-embedder` → records/sec, projected 24.2M wall-clock, cross-script recall@50. **Gate before Stage 2.**
2. `uv run pytest` → `normalize.py` per abbreviation/script/state case; `eval.py` against the PS worked example (P=2/3, R=1.0 → **0.714**); the exclusivity resolver.
3. Blocking recall ceiling + reduction ratio against train GT; ceiling should land near the measured ~99.4%.
4. `uv run ber compare-models` → stats JSON with macro F_0.5 / precision / recall / AUC-ROC for all 4 models on a held-out **entity** split.
5. Ablations: lexical-only vs. +dense; with vs. without exclusivity resolution.
6. `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` → `PASS`.
7. `matching_results.tsv` has exactly **1,732,544** data rows — one per test S1 entity, France included.
8. `docker compose up embedder` + smoke embed; `docker compose run trainer` on the 1M-pair config.

## 5. Risks

- **Embedding wall-clock** is the schedule risk. Stage 0 turns it into a measured number before
  commitment; the 1M-pair config and MRL 256d are the levers if it overruns.
- **France generalization** — no training labels exist. Mitigated by country-agnostic features, an
  entity-level validation split, and French abbreviation coverage in Stage 1.
- **87M inference pairs** — feature extraction, not model scoring, is the bottleneck. It must stay
  vectorized and chunked so pairs are never all materialized in 36 GB.

## 6. Measured results

All numbers below are from a **20,000-entity subset** (`scripts/make_subset.py`) with the
**dense channel disabled**, so they validate the pipeline rather than represent the final
score. The full run needs the 24.2M-record embedding pass first.

### Blocking (lexical channels only)

| Metric | Value |
|---|---|
| Pair recall @ K=50 | **99.21%** |
| Macro entity recall | 99.24% |
| Candidates per entity | 50.0 |
| Wall clock | 62 s for 20k entities over a 469k comparable pool |

Channel weights were tuned by sweep (4k entities, 714k pool):

| Configuration | Pair recall @ K=50 |
|---|---|
| flat weights (1,1,1,1) | 97.2% |
| tuned, no trigram channel | 98.7% |
| **tuned + trigram channel (shipped)** | **99.0%** |
| over-boosted digits (d=3.0) | 92.5% |

Remaining misses are **65% non-Latin names** and 12% empty-address records — precisely
what the dense channel is built to rescue, which is the evidence that the hybrid design
is aimed correctly.

### Models (subset, lexical-only, 1M pairs)

| Model | macro F_0.5 | threshold | precision | recall | AUC-ROC | train |
|---|---|---|---|---|---|---|
| **xgboost** | **0.98135** | 0.70 | 0.99273 | 0.96372 | 0.99990 | 20.2 s |
| lightgbm | 0.98061 | 0.70 | 0.99272 | 0.96205 | 0.99988 | 15.0 s |
| mlp | 0.97358 | 0.47 | 0.98779 | 0.95176 | 0.99751 | 11.7 s |
| entity_attn | 0.92327 | 0.05 | 0.99728 | 0.79184 | 0.98926 | 6.2 s |

Held-out split (untouched by threshold tuning), best model: **macro F_0.5 = 0.9795**,
precision 0.9921, recall 0.9641, singleton accuracy 0.951.

**`entity_attn` currently underperforms** and should not be treated as ready. Its threshold
pinned to the grid floor (0.05) and recall collapsed to 0.79 while precision hit 0.997 —
the soft-F_0.5 surrogate drove it to extreme conservatism. It ran only 4 epochs here; it
needs a proper schedule, a wider threshold grid, and probably a lower loss weight on the
singleton term before its comparison is meaningful.

Feature importance is dominated by `lex_score` (AUC 0.994), then the address family
(`addr_containment`, `addr_idf_containment`, `addr_jaccard`, all ≈0.96), then the
cross-script `skeleton_*` features (≈0.88) — matching the measured claim that address
carries the signal and name similarity is the fallback.

### Candidate contention

On the subset, **60.2% of comparable records are claimed by more than one entity**
(mean 3.2 claims, max 166). This is what the exclusivity constraint acts on. Its
measured benefit at this scale is small (+0.0002 macro F_0.5), because contested records
rarely both clear a high threshold; it should be re-measured at full scale, where each
record sits in ~8–9 candidate lists.

## 7. Environment constraints discovered

**LightGBM segfaults if PyTorch is imported first** (exit 139, in `Dataset.__init_from_np2d`).
The venv carries four OpenMP runtimes — `torch/lib/libomp.dylib`, `sklearn/.dylibs/libomp.dylib`,
`faiss/.dylibs/libomp.dylib`, and LightGBM's own `@rpath/libomp.dylib`.
`KMP_DUPLICATE_LIB_OK=TRUE` does **not** fix it. Importing LightGBM first happens to work,
but depending on import order across a package is fragile — one stray `import torch` would
reintroduce a silent crash.

The fix, in `ber/train.py` and `ber/models/registry.py`: **one model per subprocess**, with a
lazy name→class registry so no process imports a model it is not training. This also caches
features once (they are memory-mapped by each subprocess) and caps peak memory at one
model's worth instead of four.

## 8. Future scope

1. AWS deployment — the `/v1/embeddings` abstraction means swapping llama.cpp for vLLM on CUDA is
   a config change; `ChromaStore` exists for a managed vector backend.
2. Reranking with a cross-encoder over the top few candidates per entity.
3. Graph-based collective resolution across S2↔S3 (currently only S1↔S2 and S1↔S3 are modelled).
4. Fix `entity_attn` (see §6) — the sibling-competition signal is real, the current training is not.
5. Blocking misses on truncated house numbers (`1211 Cedar Grove` vs `211 Cedar Grove`) are
   unaddressed; a digit-suffix channel might recover them, at some precision cost.
