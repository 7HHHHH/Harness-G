# Harness-G: A Graph-Structured Harness for Search Agents

Harness-G trains RL search agents that navigate a **graph-structured evidence
harness** instead of writing free-form retrieval queries. At every turn the
policy chooses one action id from a discrete menu:

```text
SELECT       keep a visible sentence as evidence
LOOKUP       retrieve about a listed entity (the retrieval query is built
             automatically from the question + already-selected evidence)
ANSWER_WITH  harvest a visible sentence and finish in one step
ANSWER       stop and answer from the selected evidence
```

Training uses GRPO with **SNC (Structure-aware Navigation Credit)**: a frozen
reference policy scores the answer information gain of each navigation step,
credit is placed span-locally on the acting tokens, and enabling steps are
re-credited through provenance-dependency propagation. The outcome reward is
answer F1. See [`docs/snc_method.md`](docs/snc_method.md) for the method
description.

The code builds on [Graph-R1](https://github.com/LHRLAB/Graph-R1) and
[VERL](https://github.com/volcengine/verl).

## Final experiment protocol

The main result is a 12-run matrix: {Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct}
× six datasets, each trained and evaluated on the same dataset:

```text
2WikiMultiHopQA  HotpotQA  Musique  NQ  PopQA  TriviaQA
```

All method-side settings are **built into the code as defaults** — there are no
ablation switches. Fixed configuration:

| Component | Setting |
| --- | --- |
| Corpus | Search-R1-compatible 1,200-token chunks, 100-token overlap |
| Splits | 5,120 train / 128 dev / 128 test per dataset |
| Hardware | 8 GPUs, tensor parallel 1 |
| Rollouts / batches | 8 rollouts per prompt; train batch 128; PPO mini-batch 32 |
| Schedule | 120 steps; save every 20; dev evaluation every 10 |
| Lengths | prompt 8,192; response 2,048; tool response 4,096; ≤6 turns |
| Objective | F1 outcome reward with `algorithm.adv_estimator=grpo_snc` |
| SNC | reference scorer; span-local advantage; provenance dependencies; recursive complementarity propagation (γ=1.0); IG deadzone 1e-4; advantage scale floor 5e-4; frontier top-k 4 |
| Optimization | actor LR 5e-7; clip 0.2; grad clip 1.0; KL coefficients 0.001 |
| Numerical guards | fp32 logits; non-finite-gradient skip; dual clip 3.0 |

## Setup

```bash
conda create -n s3 python=3.9 -y
conda run -n s3 python -m pip install -e .
conda run -n s3 python -m pip install -r requirements.txt
conda run -n s3 python -m spacy download en_core_web_sm
```

The optional LLM-based G-E evaluator needs an OpenAI-compatible endpoint via
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` environment variables.
Never commit keys; `openai_api_key.txt` is git-ignored.

## Running

The versioned chunk1200 workflow lives in
[`experiments/chunk1200/`](experiments/chunk1200/README.md):

```bash
# one-time workspace setup (artifact root outside Git)
export CHUNK1200_ROOT=$HOME/harness_g_chunk1200_experiment
bash experiments/chunk1200/scripts/setup_workspace.sh

# prepare corpus + graph, then launch the 3B 8-GPU run (2WikiMultiHopQA)
conda run -n s3 python experiments/chunk1200/scripts/prepare_corpus.py
bash experiments/chunk1200/scripts/build_graph.sh
bash experiments/chunk1200/scripts/launch_full_snc.sh

# other datasets
conda run -n s3 python experiments/chunk1200/scripts/prepare_dataset_chunk1200.py --data_source HotpotQA
bash experiments/chunk1200/scripts/build_dataset_chunk1200_graph.sh HotpotQA
bash experiments/chunk1200/scripts/launch_full_snc_dataset.sh HotpotQA        # 3B
bash experiments/chunk1200/scripts/launch_full_snc_dataset_1p5b.sh HotpotQA   # 1.5B
```

The underlying training entry point is `scripts/train_harness_g_8gpu.sh`
(dataset, model, GPU topology, and paths are passed as environment variables;
everything method-related is a code default).

Tests:

```bash
conda run -n s3 python -m pytest tests/ -q
```

## Layout

```text
harness_g/                          graph index, environment, protocol, SNC credit
agent/tool/tools/harness_g_tool.py  VERL tool wrapper
verl/                               VERL fork with grpo_snc advantage + guards
scripts/run_harness_g_api.py        stateful navigation API
scripts/train_harness_g_8gpu.sh     GRPO training launcher
experiments/chunk1200/              final 12-run workflow
evaluation/                         official EM/F1/R-Sim/G-E evaluation
tests/                              regression tests
docs/snc_method.md                  SNC method description
```

Generated artifacts (`datasets/ expr/ expr_results/ runs/ checkpoints/
outputs/ wandb/`) are git-ignored.

## License

See [LICENSE](LICENSE).
