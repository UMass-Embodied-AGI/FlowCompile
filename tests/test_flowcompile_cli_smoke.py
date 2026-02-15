import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_flowcompile_compile_and_runtime_surface(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    experiment_id = "smoke_math"

    # Minimal latency profile (HF model name used by analysis mapping)
    latency_file = tmp_path / "results" / experiment_id / "01_profile" / "latency_benchmark.json"
    latency_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        latency_file,
        {
            "Qwen/Qwen3-4B": [
                {"prefill_tok_per_s": 1000.0, "decode_tok_per_s": 500.0}
            ]
        },
    )

    # Minimal trace training data
    trace_file = tmp_path / "trace_training_data.json"
    _write_json(
        trace_file,
        {
            "training_data": [
                {
                    "problem": "Solve 1+1",
                    "original_sample": {
                        "problem": "Solve 1+1",
                        "unique_id": "q1",
                    },
                }
            ]
        },
    )

    # Minimal detailed results (one setting per sub-agent)
    setting = "qwen3-4b_budget_100"
    entry = {
        "problem": "Solve 1+1",
        "accuracy": 1.0,
        "avg_input_tokens": 10,
        "avg_output_tokens": 5,
    }
    detailed_results = {
        "programmer": {setting: [entry]},
        "refine_solver": {setting: [entry]},
        "detailed_solver": {setting: [entry]},
        "generate_solver": {setting: [entry]},
        "sc_ensemble": {setting: [entry]},
    }
    detailed_file = tmp_path / "detailed_results.json"
    _write_json(detailed_file, detailed_results)

    compiled_file = tmp_path / "compiled_configs.json"
    config_file = tmp_path / "flowcompile.yaml"
    config_file.write_text(
        "\n".join(
            [
                'schema_version: "flowcompile.flat.v1"',
                f'experiment_id: "{experiment_id}"',
                'workflow_type: "math"',
                'dataset: "MATH500"',
                'model_config: "configs/config.yaml"',
                'validate_file: "data/math500_validate.jsonl"',
                'test_file: "data/math500_test.jsonl"',
                "search_axes: ['model', 'budget', 'structure']",
                "search_budgets: [100]",
                "search_models: ['qwen3-4b']",
            ]
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    )

    # Compile (predict)
    cmd = [
        sys.executable,
        "-m",
        "workflow_compiler.core.cli",
        "--config",
        str(config_file),
        "predict",
        "--workflow-type",
        "math",
        "--detailed-results",
        str(detailed_file),
        "--trace-data",
        str(trace_file),
        "--latency-file",
        f"results/{experiment_id}/01_profile/latency_benchmark.json",
        "--output-file",
        str(compiled_file),
    ]
    subprocess.run(cmd, cwd=tmp_path, check=True, env=env)

    compiled = json.loads(compiled_file.read_text())
    assert compiled.get("schema_version") == "flowcompile.compiled.v2"
    assert "configs" in compiled
    assert "levels" not in compiled

    # Removed runtime subcommands should now be rejected by parser.
    queries_file = tmp_path / "queries.jsonl"
    _write_jsonl(queries_file, [{"id": "q1", "problem": "Solve 1+1"}])

    cmd = [
        sys.executable,
        "-m",
        "workflow_compiler.core.cli",
        "runtime",
        "select",
    ]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()
