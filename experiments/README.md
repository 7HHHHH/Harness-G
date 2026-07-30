# Experiments

This directory contains the data preparation, graph construction, training,
and evaluation entry points used by Harness-G.

Runtime artifacts are stored outside the repository. Set their location before
running an experiment:

```bash
export EXPERIMENT_ROOT=/absolute/path/to/harness_g_experiments
bash experiments/scripts/setup_workspace.sh
```

## 2WikiMultiHopQA

Prepare the corpus, build the graph, and launch the 3B training recipe:

```bash
conda run -n s3 \
  python experiments/scripts/prepare_corpus.py \
  --source /absolute/path/to/2wiki_corpus.jsonl

bash experiments/scripts/build_graph.sh
bash experiments/scripts/launch_full_snc.sh
```

## Other datasets

The supported datasets are HotpotQA, MuSiQue, NQ, PopQA, and TriviaQA.

```bash
conda run -n s3 \
  python experiments/scripts/prepare_dataset.py \
  --data_source HotpotQA

bash experiments/scripts/build_dataset_graph.sh HotpotQA
bash experiments/scripts/launch_full_snc_dataset.sh HotpotQA
```

Use `launch_full_snc_dataset_1p5b.sh` for the 1.5B model. All launchers write
logs, checkpoints, and evaluation outputs beneath `$EXPERIMENT_ROOT`.

## Artifact layout

```text
$EXPERIMENT_ROOT/
├── corpus/ and graph/        2Wiki corpus and graph
├── corpora/                  prepared corpora for other datasets
├── graphs/                   dataset-specific graphs
├── datasets/                 source corpora and QA files
├── data/                     processed QA splits
├── runs/ and checkpoints/    training outputs
├── expr_results/             evaluation outputs
├── reports/ and logs/        validation reports and logs
└── workspace/                links to code and runtime artifacts
```

Generated artifacts should not be committed to Git.
