"""Helpers for staging, capturing, and validating OpenClaw Lobster workflows."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from workflow_compiler.core.llm.config import load_model_config_payload
from workflow_compiler.workflows.openclaw_lobster.parser import parse_lobster_workflow


MANIFEST_SCHEMA_VERSION = "flowcompile.openclaw.manifest.v1"
SESSION_SCHEMA_VERSION = "flowcompile.openclaw.session.v1"
VALID_JUDGE_MODES = {"strict_exact", "semantic_llm"}

_TEXT_FILE_SUFFIXES = {
    "",
    ".bash",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
    ".zsh",
}
_KNOWN_OLD_WORKSPACE_ROOTS = (
    "/home/junyan/.openclaw/workspace",
    "/root/.openclaw/workspace",
)
_KNOWN_OLD_FLOWCOMPILE_ROOTS = (
    "/home/junyan/code/FlowCompile",
    "/proj/inf-scaling/junyan/workflow_compiler/codebase",
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _normalize_env_json(raw: Optional[str]) -> Dict[str, str]:
    if raw in (None, "", "{}"):
        return {}
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid env JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("env JSON must decode to an object")
    normalized: Dict[str, str] = {}
    for key, value in payload.items():
        env_key = str(key).strip()
        if not env_key:
            raise ValueError("env JSON contains an empty variable name")
        normalized[env_key] = str(value)
    return normalized


def _normalize_args_json(raw: Optional[str]) -> str:
    if raw in (None, ""):
        return "{}"
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid args JSON: {exc}") from exc
    return json.dumps(payload, ensure_ascii=False)


def normalize_openclaw_agent_policies(raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw in (None, "", {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("openclaw_agent_policies must be a mapping of agent -> policy")

    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_agent, raw_policy in raw.items():
        agent_name = str(raw_agent or "").strip()
        if not agent_name:
            raise ValueError("openclaw_agent_policies contains an empty agent name")
        if not isinstance(raw_policy, dict):
            raise ValueError(f"openclaw_agent_policies[{agent_name!r}] must be a mapping")

        raw_required = raw_policy.get("required_fields")
        if not isinstance(raw_required, list) or not raw_required:
            raise ValueError(
                f"openclaw_agent_policies[{agent_name!r}].required_fields must be a non-empty list"
            )
        required_fields: List[str] = []
        seen = set()
        for idx, value in enumerate(raw_required):
            field_name = str(value or "").strip()
            if not field_name:
                raise ValueError(
                    f"openclaw_agent_policies[{agent_name!r}].required_fields[{idx}] must be a non-empty string"
                )
            if field_name in seen:
                raise ValueError(
                    f"openclaw_agent_policies[{agent_name!r}] contains duplicate required field {field_name!r}"
                )
            seen.add(field_name)
            required_fields.append(field_name)

        judge_cfg = raw_policy.get("judge")
        if judge_cfg is None:
            # Backward-compatible shape.
            judge_cfg = {
                "mode": raw_policy.get("mode"),
                "prompt": raw_policy.get("prompt"),
            }
        if not isinstance(judge_cfg, dict):
            raise ValueError(f"openclaw_agent_policies[{agent_name!r}].judge must be a mapping")

        mode = str(judge_cfg.get("mode") or "").strip().lower()
        if mode not in VALID_JUDGE_MODES:
            raise ValueError(
                f"openclaw_agent_policies[{agent_name!r}].judge.mode must be one of: "
                f"{', '.join(sorted(VALID_JUDGE_MODES))}"
            )

        prompt = judge_cfg.get("prompt")
        normalized_policy: Dict[str, Any] = {
            "required_fields": tuple(required_fields),
            "mode": mode,
        }
        if prompt is not None:
            prompt_text = str(prompt).strip()
            if prompt_text:
                normalized_policy["prompt"] = prompt_text
        if mode == "semantic_llm" and not normalized_policy.get("prompt"):
            raise ValueError(
                f"openclaw_agent_policies[{agent_name!r}].judge.prompt is required when mode=semantic_llm"
            )
        normalized[agent_name] = normalized_policy

    return normalized


def _detect_source_root(workflow_path: Path) -> Path:
    resolved = workflow_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        workflows_dir = candidate / "workflows"
        if workflows_dir.exists():
            try:
                resolved.relative_to(workflows_dir)
                return candidate
            except ValueError:
                pass
        if (candidate / "bin").exists() and (candidate / "prompts").exists():
            return candidate
    return resolved.parent


def _iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _TEXT_FILE_SUFFIXES:
            yield path


def _normalize_text_files(staged_root: Path, repo_root: Path) -> None:
    replacements: List[Tuple[str, str]] = [
        ("#!/home/junyan/miniconda3/bin/python", "#!/usr/bin/env python3"),
    ]
    replacements.extend((old, str(staged_root)) for old in _KNOWN_OLD_WORKSPACE_ROOTS)
    replacements.extend((old, str(repo_root)) for old in _KNOWN_OLD_FLOWCOMPILE_ROOTS)

    for path in _iter_text_files(staged_root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _inject_step_ids_into_workflow(workflow_path: Path) -> None:
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Invalid Lobster workflow (missing steps list): {workflow_path}")

    changed = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip()
        command = step.get("command")
        if not step_id or not isinstance(command, str) or not command.strip():
            continue
        if "FLOWCOMPILE_OPENCLAW_STEP_ID" in command:
            continue
        prefix = f"export FLOWCOMPILE_OPENCLAW_STEP_ID={shlex.quote(step_id)}\n"
        step["command"] = prefix + command
        changed = True

    if changed:
        workflow_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def stage_openclaw_workspace(
    workflow_file: str,
    experiment_id: str,
    *,
    source_root: Optional[str] = None,
    model_config: str = "configs/config.qwen35.local.yaml",
) -> Path:
    workflow_path = Path(workflow_file).resolve()
    if not workflow_path.exists():
        raise FileNotFoundError(f"OpenClaw workflow file not found: {workflow_path}")

    repo_root = Path.cwd().resolve()
    source_root_path = Path(source_root).resolve() if source_root else _detect_source_root(workflow_path)
    if not source_root_path.exists():
        raise FileNotFoundError(f"OpenClaw source root not found: {source_root_path}")

    try:
        workflow_rel = workflow_path.relative_to(source_root_path)
    except ValueError as exc:
        raise ValueError(
            f"workflow_file '{workflow_path}' must be contained by source_root '{source_root_path}'"
        ) from exc

    openclaw_dir = repo_root / "results" / experiment_id / "openclaw"
    staged_root = openclaw_dir / "staged_workspace"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    shutil.copytree(source_root_path, staged_root)

    _normalize_text_files(staged_root, repo_root)
    staged_workflow = staged_root / workflow_rel
    _inject_step_ids_into_workflow(staged_workflow)
    parse_lobster_workflow(str(staged_workflow))

    session_dir = openclaw_dir / "session"
    manifest_path = openclaw_dir / "manifest.json"
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "experiment_id": experiment_id,
        "repo_root": str(repo_root),
        "source_root": str(source_root_path),
        "source_workflow_file": str(workflow_path),
        "staged_root": str(staged_root),
        "staged_workflow_file": str(staged_workflow),
        "training_data_path": str(openclaw_dir / "flowcompile_training.json"),
        "analysis_path": str(openclaw_dir / "demo_analysis.json"),
        "config_path": str(openclaw_dir / "flowcompile_openclaw.yaml"),
        "session_path": str(session_dir / "session.json"),
        "capture_path": str(session_dir / "llm_capture.jsonl"),
        "model_config": model_config,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _build_node_shim(shim_path: Path, capture_path: Path, real_node_bin: str) -> None:
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    shim_code = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json
        import os
        import subprocess
        import sys
        from datetime import datetime

        def _opt(argv, name):
            try:
                i = argv.index(name)
            except ValueError:
                return None
            if i + 1 >= len(argv):
                return None
            return argv[i + 1]

        def _append_jsonl(path, payload):
            line = (json.dumps(payload, ensure_ascii=False) + "\\n").encode("utf-8")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)

        def main():
            argv = sys.argv[1:]
            real_node = os.environ.get("FLOWCOMPILE_REAL_NODE_BIN", "")
            capture_file = os.environ.get("FLOWCOMPILE_OPENCLAW_CAPTURE_FILE", "")
            step_id = os.environ.get("FLOWCOMPILE_OPENCLAW_STEP_ID", "")

            if not real_node:
                print("FLOWCOMPILE_REAL_NODE_BIN is required", file=sys.stderr)
                return 127
            if not argv:
                os.execv(real_node, [real_node])

            target = os.path.basename(argv[0])
            intercept = (
                target == "openclaw_invoke.js"
                and _opt(argv, "--tool") == "llm-task"
                and _opt(argv, "--action") == "json"
            )
            if not intercept:
                os.execv(real_node, [real_node, *argv])

            stdin_bytes = None
            if "--args-stdin" in argv:
                stdin_bytes = sys.stdin.buffer.read()

            proc = subprocess.run(
                [real_node, *argv],
                input=stdin_bytes,
                capture_output=True,
            )
            stdout_text = proc.stdout.decode("utf-8", errors="replace")
            stderr_text = proc.stderr.decode("utf-8", errors="replace")

            request_args = None
            args_file = _opt(argv, "--args-file")
            inline_args = _opt(argv, "--args-json")
            try:
                if args_file:
                    with open(args_file, "r", encoding="utf-8") as f:
                        request_args = json.load(f)
                elif "--args-stdin" in argv and stdin_bytes is not None:
                    request_args = json.loads(stdin_bytes.decode("utf-8", errors="replace"))
                elif inline_args is not None:
                    request_args = json.loads(inline_args)
            except Exception:
                request_args = None

            response_json = None
            details_json = None
            try:
                response_json = json.loads(stdout_text)
                details_json = ((response_json.get("result") or {}).get("details", {}).get("json"))
            except Exception:
                response_json = None
                details_json = None

            if capture_file:
                rec = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "argv": argv,
                    "exit_code": proc.returncode,
                    "step_id": step_id,
                    "request_args": request_args,
                    "response_stdout": stdout_text,
                    "response_stderr": stderr_text,
                    "response_json": response_json,
                    "details_json": details_json,
                }
                try:
                    _append_jsonl(capture_file, rec)
                except Exception as exc:
                    print(f"Failed to write capture log: {exc}", file=sys.stderr)

            sys.stdout.buffer.write(proc.stdout)
            sys.stderr.buffer.write(proc.stderr)
            return proc.returncode

        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
    shim_path.write_text(shim_code, encoding="utf-8")
    shim_path.chmod(0o755)


def _build_runtime_env(session_path: Path, capture_path: Path, env_overrides: Dict[str, str]) -> Dict[str, str]:
    real_node = shutil.which("node")
    if not real_node:
        raise FileNotFoundError("`node` not found in PATH")
    if not shutil.which("lobster"):
        raise FileNotFoundError("`lobster` not found in PATH")

    shim_path = session_path.parent / "bin" / "node"
    _build_node_shim(shim_path, capture_path, real_node)

    env = os.environ.copy()
    env["PATH"] = f"{shim_path.parent}{os.pathsep}{env.get('PATH', '')}"
    env["FLOWCOMPILE_REAL_NODE_BIN"] = real_node
    env["FLOWCOMPILE_OPENCLAW_CAPTURE_FILE"] = str(capture_path)
    env.update(env_overrides)
    return env


def _run_lobster_payload(cmd: Sequence[str], cwd: Path, env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse Lobster JSON output for command {' '.join(cmd)}.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Lobster command {' '.join(cmd)} returned non-object JSON payload")
    payload.setdefault("_exit_code", proc.returncode)
    payload.setdefault("_stderr", proc.stderr)
    return payload


def _resume_token(payload: Dict[str, Any]) -> Optional[str]:
    req = payload.get("requiresApproval") or {}
    token = req.get("resumeToken")
    return token if isinstance(token, str) and token else None


def _load_capture_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _render_prompt(req_args: Dict[str, Any]) -> str:
    prompt = str(req_args.get("prompt", "") or "")
    input_json = json.dumps(req_args.get("input", {}), ensure_ascii=False, indent=2, sort_keys=True)
    schema_json = json.dumps(req_args.get("schema", {}), ensure_ascii=False, indent=2, sort_keys=True)
    return f"{prompt}\n\nINPUT_JSON:\n{input_json}\n\nOUTPUT_SCHEMA_JSON:\n{schema_json}"


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _infer_agent_name_from_schema(schema: Dict[str, Any]) -> str:
    required = schema.get("required") if isinstance(schema, dict) else []
    if not isinstance(required, list):
        required = []
    required_set = {str(item) for item in required}
    if required_set == {"category"}:
        return "classify"
    if required_set == {"summary"}:
        return "summarize_each"
    if required_set == {"overview_paragraph"}:
        return "overview"
    if required_set == {"question"}:
        return "ask_questions"
    if required_set == {"draft_body"}:
        return "draft_replies"
    return "unknown"


def _build_training_data(records: List[Dict[str, Any]], run_label: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    step_number = 0
    for record in records:
        req = record.get("request_args")
        details_json = record.get("details_json")
        if not isinstance(req, dict) or details_json is None:
            continue
        schema = req.get("schema", {})
        step_number += 1
        agent_name = str(record.get("step_id") or "").strip() or _infer_agent_name_from_schema(schema if isinstance(schema, dict) else {})
        raw_out = _canonical_json_text(details_json)
        samples.append(
            {
                "agent_name": agent_name,
                "agent_input": req.get("input", {}),
                "raw_llm_prompt": _render_prompt(req),
                "raw_llm_output": raw_out,
                "processed_output": raw_out,
                "input_tokens": 0,
                "output_tokens": 0,
                "problem": run_label,
                "sample_timestamp": record.get("timestamp", _now_iso()),
                "step_number": step_number,
            }
        )
    return samples


def _finalize_training_export(manifest: Dict[str, Any], session: Dict[str, Any]) -> None:
    capture_path = Path(session["capture_path"])
    records = _load_capture_records(capture_path)
    run_label = f"{Path(manifest['staged_workflow_file']).stem}:{datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')}"
    training_data = _build_training_data(records, run_label=run_label)
    payload = {
        "metadata": {
            "generated_at": _now_iso(),
            "workflow": manifest["staged_workflow_file"],
            "run_label": run_label,
            "llm_call_count": len(training_data),
            "captured_record_count": len(records),
            "session_status": session.get("status"),
        },
        "training_data": training_data,
    }
    training_path = Path(manifest["training_data_path"])
    _write_json(training_path, payload)
    session["training_data_path"] = str(training_path)
    session["llm_call_count"] = len(training_data)
    session["captured_record_count"] = len(records)


def demo_run_openclaw(manifest_path: str, *, args_json: Optional[str] = None, env_json: Optional[str] = None) -> Path:
    manifest = _read_json(Path(manifest_path))
    session_path = Path(manifest["session_path"])
    capture_path = Path(manifest["capture_path"])
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    if capture_path.exists():
        capture_path.unlink()

    env_overrides = _normalize_env_json(env_json)
    env = _build_runtime_env(session_path, capture_path, env_overrides)
    normalized_args_json = _normalize_args_json(args_json)
    payload = _run_lobster_payload(
        [
            "lobster",
            "run",
            "--mode",
            "tool",
            "--file",
            manifest["staged_workflow_file"],
            "--args-json",
            normalized_args_json,
        ],
        cwd=Path(manifest["staged_root"]),
        env=env,
    )

    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "capture_path": str(capture_path),
        "status": payload.get("status") or "unknown",
        "args_json": json.loads(normalized_args_json),
        "env_overrides": env_overrides,
        "resume_token": _resume_token(payload),
        "last_payload": payload,
    }
    if payload.get("status") == "ok":
        session["status"] = "completed"
        _finalize_training_export(manifest, session)
    elif payload.get("status") not in {"needs_approval", "ok"}:
        session["status"] = "failed"
    _write_json(session_path, session)
    return session_path


def demo_resume_openclaw(
    session_path: str,
    *,
    approve: str = "yes",
    env_json: Optional[str] = None,
) -> Path:
    session_file = Path(session_path)
    session = _read_json(session_file)
    manifest = _read_json(Path(session["manifest_path"]))
    resume_token = str(session.get("resume_token") or "").strip()
    if not resume_token:
        raise ValueError(f"Session does not contain a resume token: {session_file}")

    env_overrides = dict(session.get("env_overrides") or {})
    env_overrides.update(_normalize_env_json(env_json))
    env = _build_runtime_env(session_file, Path(session["capture_path"]), env_overrides)
    payload = _run_lobster_payload(
        [
            "lobster",
            "resume",
            "--mode",
            "tool",
            "--token",
            resume_token,
            "--approve",
            str(approve or "yes"),
        ],
        cwd=Path(manifest["staged_root"]),
        env=env,
    )
    session.update(
        {
            "updated_at": _now_iso(),
            "status": payload.get("status") or "unknown",
            "resume_token": _resume_token(payload),
            "last_payload": payload,
            "env_overrides": env_overrides,
        }
    )
    if payload.get("status") == "ok":
        session["status"] = "completed"
        _finalize_training_export(manifest, session)
    elif payload.get("status") not in {"needs_approval", "ok"}:
        session["status"] = "failed"
    _write_json(session_file, session)
    return session_file


def _extract_required_fields(samples: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    parsed_objects: List[Dict[str, Any]] = []
    for sample in samples:
        try:
            payload = json.loads(str(sample.get("processed_output") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed_objects.append(payload)
    if not parsed_objects:
        return [], [], {}

    union_fields = sorted({key for payload in parsed_objects for key in payload.keys()})
    intersection = sorted(set(parsed_objects[0].keys()).intersection(*(payload.keys() for payload in parsed_objects[1:])))
    field_types: Dict[str, List[str]] = {}
    for field_name in union_fields:
        types = sorted({type(payload.get(field_name)).__name__ for payload in parsed_objects if field_name in payload})
        field_types[field_name] = types
    return intersection, union_fields, field_types


def _agent_upstreams(spec: Dict[str, Any], node_id: str) -> List[str]:
    node_by_id = {str(node.get("id")): node for node in (spec.get("nodes") or []) if node.get("id")}
    node = node_by_id.get(node_id)
    if not isinstance(node, dict):
        return []
    io = node.get("io") or {}
    inputs = io.get("inputs") if isinstance(io, dict) else {}
    refs: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("ref")
            if isinstance(ref, str) and ref.startswith("state."):
                refs.append(ref.split(".", 1)[1])
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    walk(inputs)
    seen = set()
    ordered: List[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered


def infer_candidate_workflow_loops(spec: Dict[str, Any], counts_by_agent: Dict[str, int]) -> List[Dict[str, Any]]:
    nodes = [node for node in (spec.get("nodes") or []) if node.get("type") == "agent"]
    order = [str(node.get("id")) for node in nodes if node.get("id")]
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    assigned = set()
    loops: List[Dict[str, Any]] = []

    for node_id in order:
        node = node_by_id[node_id]
        operator = str((node.get("metadata") or {}).get("operator") or "").lower()
        if operator not in {"map_reduce", "map-reduce", "reduce"}:
            continue
        upstreams = _agent_upstreams(spec, node_id)
        if not upstreams:
            continue
        loop_count = max(int(counts_by_agent.get(upstream, 0) or 0) for upstream in upstreams)
        if loop_count <= 1:
            continue
        loops.append(
            {
                "name": f"{node_id}_loop",
                "count": loop_count,
                "map_nodes": upstreams,
                "reduce_node": node_id,
                "observed_counts": {agent: int(counts_by_agent.get(agent, 0) or 0) for agent in [*upstreams, node_id]},
            }
        )
        assigned.update(upstreams)
        assigned.add(node_id)

    idx = 0
    while idx < len(order):
        node_id = order[idx]
        loop_count = int(counts_by_agent.get(node_id, 0) or 0)
        if node_id in assigned or loop_count <= 1:
            idx += 1
            continue

        chain = [node_id]
        next_idx = idx + 1
        while next_idx < len(order):
            candidate = order[next_idx]
            candidate_count = int(counts_by_agent.get(candidate, 0) or 0)
            upstreams = set(_agent_upstreams(spec, candidate))
            if candidate in assigned or candidate_count != loop_count or not (upstreams & set(chain)):
                break
            chain.append(candidate)
            next_idx += 1

        loops.append(
            {
                "name": f"{chain[0]}_loop",
                "count": loop_count,
                "map_nodes": chain,
                "observed_counts": {agent: int(counts_by_agent.get(agent, 0) or 0) for agent in chain},
            }
        )
        assigned.update(chain)
        idx = next_idx

    return loops


def analyze_openclaw_demo(manifest_path: str) -> Path:
    manifest = _read_json(Path(manifest_path))
    training_payload = _read_json(Path(manifest["training_data_path"]))
    training_data = training_payload.get("training_data") or []
    if not isinstance(training_data, list):
        raise ValueError(f"training_data must be a list in {manifest['training_data_path']}")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in training_data:
        if not isinstance(sample, dict):
            continue
        agent_name = str(sample.get("agent_name") or "").strip()
        if not agent_name:
            continue
        grouped.setdefault(agent_name, []).append(sample)

    spec = parse_lobster_workflow(manifest["staged_workflow_file"])
    counts_by_agent = {agent_name: len(samples) for agent_name, samples in grouped.items()}
    agents: Dict[str, Dict[str, Any]] = {}
    for agent_name, samples in sorted(grouped.items()):
        required_fields, observed_fields, field_types = _extract_required_fields(samples)
        examples = []
        for sample in samples[:3]:
            examples.append(
                {
                    "raw_llm_prompt": str(sample.get("raw_llm_prompt") or "")[:1200],
                    "processed_output": sample.get("processed_output"),
                    "raw_llm_output": sample.get("raw_llm_output"),
                }
            )
        agents[agent_name] = {
            "sample_count": len(samples),
            "required_fields_intersection": required_fields,
            "observed_fields_union": observed_fields,
            "observed_field_types": field_types,
            "examples": examples,
        }

    openclaw_dir = Path(manifest["training_data_path"]).parent
    analysis_path = Path(manifest["analysis_path"])
    payload = {
        "schema_version": "flowcompile.openclaw.analysis.v1",
        "generated_at": _now_iso(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "workflow": {
            "name": spec.get("name"),
            "llm_steps": [
                {
                    "id": str(node.get("id")),
                    "operator": str((node.get("metadata") or {}).get("operator") or ""),
                    "upstream_llm_steps": _agent_upstreams(spec, str(node.get("id"))),
                }
                for node in (spec.get("nodes") or [])
                if node.get("type") == "agent"
            ],
        },
        "training_data_summary": {
            "llm_call_count": len(training_data),
            "counts_by_agent": counts_by_agent,
        },
        "agents": agents,
        "candidate_workflow_loops": infer_candidate_workflow_loops(spec, counts_by_agent),
        "config_authoring": {
            "suggested_config_path": os.path.relpath(Path(manifest["config_path"]), openclaw_dir),
            "relative_paths": {
                "openclaw_lobster_workflow_file": os.path.relpath(Path(manifest["staged_workflow_file"]), openclaw_dir),
                "profile_training_data": os.path.relpath(Path(manifest["training_data_path"]), openclaw_dir),
                "predict_trace_data": os.path.relpath(Path(manifest["training_data_path"]), openclaw_dir),
                "model_config": os.path.relpath(Path(manifest["repo_root"]) / manifest["model_config"], openclaw_dir),
            },
            "required_keys": [
                "schema_version",
                "experiment_id",
                "workflow_type",
                "model_config",
                "openclaw_lobster_workflow_file",
                "profile_training_data",
                "predict_trace_data",
                "search_axes",
                "search_models",
                "search_budgets",
                "profile_models",
                "latency_models",
            ],
            "optional_keys": [
                "openclaw_agent_policies",
                "workflow_loops",
                "predict_subagent_score_thresholds",
                "profile_min_samples_per_agent",
                "profile_max_concurrent",
            ],
        },
    }
    _write_json(analysis_path, payload)
    return analysis_path


def _validate_workflow_loops_against_spec(spec: Dict[str, Any], raw_loops: Any) -> None:
    if raw_loops in (None, "", []):
        return
    if not isinstance(raw_loops, list):
        raise ValueError("workflow_loops must be a list of loop definitions")

    node_by_id = {
        str(node.get("id")): node
        for node in (spec.get("nodes") or [])
        if node.get("type") == "agent" and node.get("id")
    }
    assigned: Dict[str, str] = {}
    seen_names = set()
    for idx, item in enumerate(raw_loops):
        if not isinstance(item, dict):
            raise ValueError(f"workflow_loops[{idx}] must be a mapping")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"workflow_loops[{idx}].name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"workflow_loops contains duplicate loop name '{name}'")
        seen_names.add(name)
        count = item.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"workflow_loops[{idx}].count must be an integer >= 1")

        map_nodes = item.get("map_nodes")
        if not isinstance(map_nodes, list) or not map_nodes:
            raise ValueError(f"workflow_loops[{idx}].map_nodes must be a non-empty list")
        local_seen = set()
        for map_idx, raw_node in enumerate(map_nodes):
            node_id = str(raw_node or "").strip()
            if not node_id:
                raise ValueError(f"workflow_loops[{idx}].map_nodes[{map_idx}] must be a non-empty string")
            if node_id in local_seen:
                raise ValueError(f"workflow_loops[{idx}] contains duplicate node '{node_id}' in map_nodes")
            if node_id not in node_by_id:
                raise ValueError(
                    f"workflow_loops loop '{name}' references unknown or inactive map node '{node_id}'"
                )
            owner = assigned.get(node_id)
            if owner is not None:
                raise ValueError(f"workflow_loops node '{node_id}' is assigned to both '{owner}' and '{name}'")
            assigned[node_id] = name
            local_seen.add(node_id)

        reduce_raw = item.get("reduce_node")
        if reduce_raw is not None:
            reduce_node = str(reduce_raw or "").strip()
            if not reduce_node:
                raise ValueError(f"workflow_loops[{idx}].reduce_node must be a non-empty string")
            if reduce_node in local_seen:
                raise ValueError(
                    f"workflow_loops[{idx}].reduce_node '{reduce_node}' cannot also appear in map_nodes"
                )
            if reduce_node not in node_by_id:
                raise ValueError(
                    f"workflow_loops loop '{name}' references unknown or inactive reduce node '{reduce_node}'"
                )
            owner = assigned.get(reduce_node)
            if owner is not None:
                raise ValueError(
                    f"workflow_loops node '{reduce_node}' is assigned to both '{owner}' and '{name}'"
                )
            operator = str((node_by_id[reduce_node].get("metadata") or {}).get("operator") or "").lower()
            if operator not in {"map_reduce", "map-reduce", "reduce"}:
                raise ValueError(
                    f"workflow_loops loop '{name}' reduce node '{reduce_node}' "
                    f"must use operator map_reduce or reduce, found '{operator}'"
                )
            assigned[reduce_node] = name


def _validate_thresholds(raw: Any, valid_agents: Sequence[str]) -> None:
    if raw in (None, "", {}):
        return
    if not isinstance(raw, dict):
        raise ValueError("predict_subagent_score_thresholds must be a mapping of subagent -> threshold")
    valid_set = set(valid_agents)
    for raw_agent, raw_threshold in raw.items():
        agent_name = str(raw_agent or "").strip()
        if not agent_name:
            raise ValueError("predict_subagent_score_thresholds contains an empty subagent name")
        if agent_name not in valid_set:
            raise ValueError(
                "predict_subagent_score_thresholds contains unknown subagent "
                f"{agent_name!r}. Valid subagents: {sorted(valid_set)}"
            )
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"predict_subagent_score_thresholds[{agent_name!r}] must be a float in [0.0, 1.0]"
            ) from exc
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                f"predict_subagent_score_thresholds[{agent_name!r}] must be a float in [0.0, 1.0]"
            )


def validate_openclaw_config_payload(cfg: Dict[str, Any], *, config_path: Optional[str] = None) -> Dict[str, Any]:
    workflow_type = str(cfg.get("workflow_type") or "").strip().lower()
    if workflow_type != "openclaw_lobster":
        raise ValueError("validate-config only supports workflow_type=openclaw_lobster")

    for key in ("search_axes", "search_budgets", "search_models", "profile_models", "latency_models"):
        value = cfg.get(key)
        if value in (None, "", []):
            raise ValueError(f"{key} is required for openclaw_lobster configs")

    workflow_file = cfg.get("openclaw_lobster_workflow_file")
    if not workflow_file:
        raise ValueError("openclaw_lobster_workflow_file is required")
    training_data = cfg.get("profile_training_data")
    if not training_data:
        raise ValueError("profile_training_data is required")
    trace_data = cfg.get("predict_trace_data")
    if not trace_data:
        raise ValueError("predict_trace_data is required")

    workflow_path = Path(str(workflow_file))
    training_path = Path(str(training_data))
    trace_path = Path(str(trace_data))
    for label, path in (
        ("openclaw_lobster_workflow_file", workflow_path),
        ("profile_training_data", training_path),
        ("predict_trace_data", trace_path),
    ):
        if not path.exists():
            raise ValueError(f"{label} not found: {path}")

    model_config = cfg.get("model_config")
    if not model_config:
        raise ValueError("model_config is required")
    try:
        load_model_config_payload(model_config)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    normalized_policies = normalize_openclaw_agent_policies(cfg.get("openclaw_agent_policies"))
    if not normalized_policies:
        raise ValueError("openclaw_agent_policies is required for openclaw_lobster profiling")

    spec = parse_lobster_workflow(str(workflow_path))
    agent_ids = [str(node.get("id")) for node in (spec.get("nodes") or []) if node.get("type") == "agent" and node.get("id")]
    missing = sorted(set(agent_ids) - set(normalized_policies.keys()))
    unknown = sorted(set(normalized_policies.keys()) - set(agent_ids))
    if missing:
        raise ValueError(f"openclaw_agent_policies is missing workflow agents: {missing}")
    if unknown:
        raise ValueError(f"openclaw_agent_policies contains unknown workflow agents: {unknown}")

    _validate_workflow_loops_against_spec(spec, cfg.get("workflow_loops"))
    _validate_thresholds(cfg.get("predict_subagent_score_thresholds"), agent_ids)

    summary = {
        "workflow_agents": agent_ids,
        "normalized_policies": normalized_policies,
        "config_path": config_path,
    }
    return summary


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SESSION_SCHEMA_VERSION",
    "normalize_openclaw_agent_policies",
    "stage_openclaw_workspace",
    "demo_run_openclaw",
    "demo_resume_openclaw",
    "analyze_openclaw_demo",
    "infer_candidate_workflow_loops",
    "validate_openclaw_config_payload",
]
