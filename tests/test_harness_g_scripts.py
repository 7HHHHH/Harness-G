import os
import subprocess
import sys
from pathlib import Path

from script_process_harness_g import INSTRUCTION, _supporting_evidence_from_example
from omegaconf import OmegaConf

from verl.trainer.main_ppo import _ensure_harness_g_env, _harness_g_runtime_env_vars


CORPUS = """{"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London. She worked with Charles Babbage on the Analytical Engine."}
{"id": "2", "title": "Charles Babbage", "contents": "Charles Babbage designed the Analytical Engine. He was born in London."}
"""


def test_build_and_train_script_dry_run_fail_fast(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/build_harness_g_graph.py",
            "--corpus_path",
            str(corpus_path),
            "--output_dir",
            str(graph_dir),
        ],
        check=True,
    )
    subprocess.run([sys.executable, "scripts/validate_harness_g_graph.py", "--graph_dir", str(graph_dir)], check=True)
    assert Path("scripts/setup_harness_g_conda.sh").exists()
    assert Path("scripts/train_harness_g_8gpu.sh").exists()


def test_train_harness_g_8gpu_dry_run(tmp_path):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            "bash",
            "scripts/train_harness_g_8gpu.sh",
        ],
        env={
            **dict(os.environ),
            "ENV_NAME": "s3",
            "DRY_RUN": "true",
            "RUN_DIR": str(run_dir),
            # Isolate the canonical 2Wiki dry-run from an outer dataset batch.
            "DATA_SOURCE": "2WikiMultiHopQA",
            "PROCESSED_DIR": "datasets/2WikiMultiHopQA/processed",
            "VAL_SPLIT": "dev",
            "PYTEST_TARGETS": "tests/test_harness_g_protocol.py",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "N_GPUS": "4",
            "TP_SIZE": "1",
            "ROLLOUT_N": "8",
            "TRAIN_BATCH_SIZE": "64",
            "PPO_MINI_BATCH_SIZE": "32",
            "MAX_TURNS": "6",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "DRY_RUN=true" in result.stdout
    command = (run_dir / "train_command.txt").read_text(encoding="utf-8")
    assert "algorithm.adv_estimator=grpo_snc" in command
    assert "data.val_files=datasets/2WikiMultiHopQA/processed/dev.parquet" in command
    assert "trainer.n_gpus_per_node=4" in command
    assert "actor_rollout_ref.rollout.n_repeat=8" in command
    assert "tool.max_turns=6" in command
    assert (run_dir / "run_config.json").exists()
    run_config = (run_dir / "run_config.json").read_text(encoding="utf-8")
    assert '"validation_split": "dev"' in run_config
    assert '"adv_estimator": "grpo_snc"' in run_config
    assert '"reward": "f1_outcome"' in run_config


def test_harness_g_processing_prompt_mentions_new_actions():
    assert "SELECT" in INSTRUCTION
    assert "LOOKUP" in INSTRUCTION
    assert "ANSWER_WITH" in INSTRUCTION
    for legacy in ("OPEN_CONTEXT", "BRIDGE_ENTITY", "EXPAND_ENTITY", "REWRITE_QUERY", "STOP"):
        assert legacy not in INSTRUCTION
    assert _supporting_evidence_from_example({"supporting_facts": ["Ada evidence"]}) == ["Ada evidence"]


def test_harness_g_ray_runtime_env_propagates_reward_vars(monkeypatch):
    monkeypatch.setenv("HARNESS_G_REWARD", "1")
    monkeypatch.setenv("HARNESS_G_REWARD_METRICS_PATH", "runs/test/reward_metrics.jsonl")
    monkeypatch.setenv("HARNESS_G_API_URL", "http://localhost:8001/harness_g_step")
    env_vars = _harness_g_runtime_env_vars()
    assert env_vars["HARNESS_G_REWARD"] == "1"
    assert env_vars["HARNESS_G_REWARD_METRICS_PATH"] == "runs/test/reward_metrics.jsonl"
    assert env_vars["HARNESS_G_API_URL"] == "http://localhost:8001/harness_g_step"


def test_harness_g_main_task_forces_reward_env(monkeypatch):
    monkeypatch.delenv("HARNESS_G_REWARD", raising=False)
    monkeypatch.delenv("HARNESS_G_RUN_DIR", raising=False)
    monkeypatch.delenv("HARNESS_G_REWARD_METRICS_PATH", raising=False)
    config = OmegaConf.create({"tool": {"env": "harness_g"}, "trainer": {"experiment_name": "harness_g_env_test"}})
    _ensure_harness_g_env(config)
    assert os.environ["HARNESS_G_REWARD"] == "1"
    assert os.environ["HARNESS_G_RUN_DIR"] == "runs/harness_g_env_test"
    assert os.environ["HARNESS_G_REWARD_METRICS_PATH"] == "runs/harness_g_env_test/reward_metrics.jsonl"
