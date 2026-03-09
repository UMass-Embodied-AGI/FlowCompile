import json

from workflow_compiler.compiler import latency


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_save_latency_json_merges_existing_models(tmp_path):
    output = tmp_path / "latency_benchmark.json"
    output.write_text(
        json.dumps(
            {
                "Qwen/Qwen3-4B": [{"batch_size": 1, "prefill_tok_per_s": 1000, "decode_tok_per_s": 500}],
            }
        ),
        encoding="utf-8",
    )

    latency._save_latency_json(
        str(output),
        {
            "Qwen/Qwen3-8B": [{"batch_size": 1, "prefill_tok_per_s": 800, "decode_tok_per_s": 300}],
        },
    )

    saved = _read_json(output)
    assert set(saved.keys()) == {"Qwen/Qwen3-4B", "Qwen/Qwen3-8B"}
    assert saved["Qwen/Qwen3-4B"][0]["prefill_tok_per_s"] == 1000
    assert saved["Qwen/Qwen3-8B"][0]["decode_tok_per_s"] == 300


def test_save_latency_json_upserts_existing_model(tmp_path):
    output = tmp_path / "latency_benchmark.json"
    output.write_text(
        json.dumps(
            {
                "Qwen/Qwen3-4B": [{"batch_size": 1, "prefill_tok_per_s": 1000, "decode_tok_per_s": 500}],
            }
        ),
        encoding="utf-8",
    )

    latency._save_latency_json(
        str(output),
        {
            "Qwen/Qwen3-4B": [{"batch_size": 2, "prefill_tok_per_s": 1200, "decode_tok_per_s": 600}],
        },
    )

    saved = _read_json(output)
    assert set(saved.keys()) == {"Qwen/Qwen3-4B"}
    assert saved["Qwen/Qwen3-4B"][0]["batch_size"] == 2
    assert saved["Qwen/Qwen3-4B"][0]["prefill_tok_per_s"] == 1200
