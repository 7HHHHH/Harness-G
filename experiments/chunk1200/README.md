# Canonical chunk1200 workflow

Graph-R1 now treats the Search-R1-compatible 1,200-token / 100-token-overlap
corpus as the canonical experiment setting. Versioned orchestration lives in
this directory and always executes the current repository checkout.

Large runtime artifacts remain outside Git by default:

```bash
export CHUNK1200_ROOT=$HOME/harness_g_chunk1200_experiment
```

The existing artifact directory is intentionally retained because it contains
the prepared corpora, graphs, checkpoints, runs, and evaluation results. Its
legacy `code/Graph-R1` snapshot is no longer an execution source; after setup,
`workspace/` points to the current `main` checkout instead.

## First-time setup or migration

From the Graph-R1 repository root:

```bash
bash experiments/chunk1200/scripts/setup_workspace.sh
```

The setup command is idempotent. It only replaces managed symlinks inside
`$CHUNK1200_ROOT/workspace`; it refuses to overwrite real files or directories.

## 2WikiMultiHopQA

The frozen source is Search-R1's 2,811-record chunk1200 corpus. Preparation
normalizes whitespace so Graph-R1's packed-corpus loader cannot reinterpret
quoted title lines as document boundaries.

```bash
conda run -n s3 \
  python experiments/chunk1200/scripts/prepare_corpus.py
bash experiments/chunk1200/scripts/build_graph.sh
bash experiments/chunk1200/scripts/launch_full_snc.sh
```

`launch_full_snc.sh` is the canonical 3B, eight-GPU recipe.

## Other datasets

Prepare and build one dataset-specific chunk1200 graph:

```bash
conda run -n s3 \
  python experiments/chunk1200/scripts/prepare_dataset_chunk1200.py \
  --data_source HotpotQA
bash experiments/chunk1200/scripts/build_dataset_chunk1200_graph.sh HotpotQA
```

Launch 3B or 1.5B full SNC:

```bash
bash experiments/chunk1200/scripts/launch_full_snc_dataset.sh HotpotQA
bash experiments/chunk1200/scripts/launch_full_snc_dataset_1p5b.sh HotpotQA
```

Launchers are detached and write into `$CHUNK1200_ROOT`. Override the artifact
location with `CHUNK1200_ROOT`, and override the code checkout with `REPO_ROOT`
only when deliberately testing another tree.

## Artifact layout

```text
$CHUNK1200_ROOT/
├── corpus/ and graph/            2Wiki frozen corpus and graph
├── corpora_chunk1200/            per-dataset prepared corpora
├── graphs_chunk1200/             per-dataset graphs
├── data/                         processed QA splits
├── runs/ and checkpoints/        training outputs
├── expr_results/                 evaluation outputs
├── reports/                      validation reports
├── logs/                         batch-run logs
└── workspace/                    symlinks to current main + artifact dirs
```

Do not add these generated artifacts to Git. Only the scripts and documentation
under `experiments/chunk1200/` are versioned.
