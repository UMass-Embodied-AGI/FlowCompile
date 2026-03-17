"""Lobster YAML -> FlowCompile workflow spec parser."""
from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Set

import yaml


_STEP_REF_RE = re.compile(r"\$([A-Za-z0-9_-]+)\.stdout")
_BatchKind = Literal["single", "multi", "unknown"]


@dataclass(frozen=True)
class _StepScriptInfo:
    path: Path
    artifact_reads: tuple[str, ...]
    llm_agent: str | None
    batch_kind: _BatchKind | None
    output_schema_path: Path | None


def _extract_step_refs(text: Any) -> Set[str]:
    if not isinstance(text, str) or not text:
        return set()
    return set(_STEP_REF_RE.findall(text))


def _resolve_python_script(bundle_dir: Path, command: Any) -> Path | None:
    if not isinstance(command, str) or not command.strip():
        return None

    for line in command.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            continue
        for token in tokens:
            if not token.endswith(".py"):
                continue
            candidate = Path(token)
            if not candidate.is_absolute():
                candidate = (bundle_dir / candidate).resolve()
            if candidate.exists():
                return candidate
    return None


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _build_parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _classify_list_literal(node: ast.List) -> _BatchKind:
    if len(node.elts) == 1:
        return "single"
    if len(node.elts) > 1:
        return "multi"
    return "unknown"


def _classify_items_expr(node: ast.AST) -> _BatchKind:
    if isinstance(node, ast.List):
        return _classify_list_literal(node)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return "multi"
    return "unknown"


def _enclosing_scope(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> ast.AST:
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return cur
    return cur


def _module_assignments(tree: ast.Module) -> Dict[str, ast.AST]:
    assignments: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    return assignments


def _latest_named_value(var_name: str, scope: ast.AST, before_lineno: int) -> ast.AST | None:
    latest_assignment: ast.AST | None = None
    latest_assignment_lineno = -1

    for node in ast.walk(scope):
        lineno = getattr(node, "lineno", None)
        if lineno is None or lineno >= before_lineno:
            continue

        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        if any(isinstance(target, ast.Name) and target.id == var_name for target in targets):
            if lineno >= latest_assignment_lineno:
                latest_assignment = value
                latest_assignment_lineno = lineno

    return latest_assignment


def _infer_named_items_batch_kind(var_name: str, scope: ast.AST, call_lineno: int) -> _BatchKind:
    saw_multi_mutation = False

    for node in ast.walk(scope):
        lineno = getattr(node, "lineno", None)
        if lineno is None or lineno >= call_lineno:
            continue

        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == var_name:
            if isinstance(node.op, ast.Add):
                saw_multi_mutation = True
            continue

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == var_name and node.func.attr in {"append", "extend", "insert"}:
                saw_multi_mutation = True

    if saw_multi_mutation:
        return "multi"
    latest_assignment = _latest_named_value(var_name, scope, call_lineno)
    if latest_assignment is None:
        return "unknown"
    return _classify_items_expr(latest_assignment)


def _infer_run_json_batch_kind(call: ast.Call, parents: Dict[ast.AST, ast.AST]) -> _BatchKind:
    items_value: ast.AST | None = None
    for kw in call.keywords:
        if kw.arg == "items":
            items_value = kw.value
            break
    if items_value is None:
        return "unknown"
    if isinstance(items_value, ast.Name):
        scope = _enclosing_scope(call, parents)
        return _infer_named_items_batch_kind(items_value.id, scope, getattr(call, "lineno", 0))
    return _classify_items_expr(items_value)


def _load_object_schema(schema_path: Path, *, script_path: Path, step_id: str) -> Dict[str, Any]:
    if not schema_path.exists():
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} references missing schema file {schema_path} "
            f"in {script_path}"
        )
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} references invalid JSON schema {schema_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} schema must decode to an object: {schema_path}"
        )
    if str(payload.get("type") or "").strip().lower() != "object":
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} schema must declare root type 'object': {schema_path}"
        )
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} schema must define an object 'properties' mapping: "
            f"{schema_path}"
        )
    return payload


def _resolve_path_expr(
    node: ast.AST,
    *,
    script_path: Path,
    bundle_dir: Path,
    module_assignments: Dict[str, ast.AST],
    scope: ast.AST,
    call_lineno: int,
    seen_names: Set[str],
) -> Path | None:
    if isinstance(node, ast.Name):
        if node.id in seen_names:
            return None
        seen_names.add(node.id)
        value = _latest_named_value(node.id, scope, call_lineno)
        if value is None:
            value = module_assignments.get(node.id)
        if value is None:
            return None
        return _resolve_path_expr(
            value,
            script_path=script_path,
            bundle_dir=bundle_dir,
            module_assignments=module_assignments,
            scope=scope,
            call_lineno=call_lineno,
            seen_names=seen_names,
        )

    literal = _literal_str(node)
    if literal is not None:
        candidate = Path(literal)
        if candidate.is_absolute():
            return candidate.resolve()
        return (script_path.parent / candidate).resolve()

    if isinstance(node, ast.Call):
        call_name = _call_name(node)
        if call_name == "Path" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id == "__file__":
                return script_path.resolve()
            literal_arg = _literal_str(arg)
            if literal_arg is not None:
                candidate = Path(literal_arg)
                if candidate.is_absolute():
                    return candidate.resolve()
                return (script_path.parent / candidate).resolve()
        if call_name == "schema_path" and node.args:
            schema_name = _literal_str(node.args[0])
            if schema_name:
                return (bundle_dir / "prompts" / f"{schema_name}.schema.json").resolve()
        if isinstance(node.func, ast.Attribute):
            base = _resolve_path_expr(
                node.func.value,
                script_path=script_path,
                bundle_dir=bundle_dir,
                module_assignments=module_assignments,
                scope=scope,
                call_lineno=call_lineno,
                seen_names=seen_names,
            )
            if base is None:
                return None
            if node.func.attr in {"resolve", "absolute"}:
                return base.resolve()
            if node.func.attr == "joinpath":
                current = base
                for arg in node.args:
                    part = _literal_str(arg)
                    if part is None:
                        return None
                    current = current / part
                return current.resolve()
        return None

    if isinstance(node, ast.Attribute):
        base = _resolve_path_expr(
            node.value,
            script_path=script_path,
            bundle_dir=bundle_dir,
            module_assignments=module_assignments,
            scope=scope,
            call_lineno=call_lineno,
            seen_names=seen_names,
        )
        if base is None:
            return None
        if node.attr == "parent":
            return base.parent
        return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve_path_expr(
            node.left,
            script_path=script_path,
            bundle_dir=bundle_dir,
            module_assignments=module_assignments,
            scope=scope,
            call_lineno=call_lineno,
            seen_names=seen_names,
        )
        right = _literal_str(node.right)
        if left is None or right is None:
            return None
        return (left / right).resolve()

    return None


def _resolve_schema_path_expr(
    node: ast.AST,
    *,
    script_path: Path,
    bundle_dir: Path,
    module_assignments: Dict[str, ast.AST],
    scope: ast.AST,
    call_lineno: int,
    seen_names: Set[str],
) -> Path | None:
    if isinstance(node, ast.Call):
        call_name = _call_name(node)
        if call_name in {"load_schema", "schema_path"} and node.args:
            schema_name = _literal_str(node.args[0])
            if schema_name:
                return (bundle_dir / "prompts" / f"{schema_name}.schema.json").resolve()
        if call_name == "loads" and node.args:
            return _resolve_schema_path_expr(
                node.args[0],
                script_path=script_path,
                bundle_dir=bundle_dir,
                module_assignments=module_assignments,
                scope=scope,
                call_lineno=call_lineno,
                seen_names=seen_names,
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
            return _resolve_path_expr(
                node.func.value,
                script_path=script_path,
                bundle_dir=bundle_dir,
                module_assignments=module_assignments,
                scope=scope,
                call_lineno=call_lineno,
                seen_names=seen_names,
            )

    if isinstance(node, ast.Name):
        if node.id in seen_names:
            return None
        seen_names.add(node.id)
        value = _latest_named_value(node.id, scope, call_lineno)
        if value is None:
            value = module_assignments.get(node.id)
        if value is None:
            return None
        return _resolve_schema_path_expr(
            value,
            script_path=script_path,
            bundle_dir=bundle_dir,
            module_assignments=module_assignments,
            scope=scope,
            call_lineno=call_lineno,
            seen_names=seen_names,
        )

    return _resolve_path_expr(
        node,
        script_path=script_path,
        bundle_dir=bundle_dir,
        module_assignments=module_assignments,
        scope=scope,
        call_lineno=call_lineno,
        seen_names=seen_names,
    )


def _inspect_python_script(step_id: str, script_path: Path) -> _StepScriptInfo:
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse Python step script {script_path}: {exc}") from exc

    parents = _build_parent_map(tree)
    module_assignments = _module_assignments(tree)
    bundle_dir = script_path.parent.parent
    artifact_reads: Set[str] = set()
    llm_agents: Set[str] = set()
    batch_kinds: Set[_BatchKind] = set()
    schema_paths: Set[Path] = set()
    saw_nonliteral_agent = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        call_name = _call_name(node)
        if call_name == "read_artifact" and len(node.args) >= 2:
            artifact_name = _literal_str(node.args[1])
            if artifact_name:
                artifact_reads.add(artifact_name)

        if call_name != "run_json_batch":
            continue

        agent_value = None
        for kw in node.keywords:
            if kw.arg == "agent":
                agent_value = _literal_str(kw.value)
                if agent_value is None:
                    saw_nonliteral_agent = True
                break
        if agent_value is not None:
            llm_agents.add(agent_value)
        batch_kinds.add(_infer_run_json_batch_kind(node, parents))
        scope = _enclosing_scope(node, parents)
        schema_value = next((kw.value for kw in node.keywords if kw.arg == "schema"), None)
        if schema_value is None:
            raise ValueError(
                f"OpenClaw workflow step {step_id!r} must pass a statically discoverable schema=... "
                f"argument to run_json_batch in {script_path}"
            )
        schema_path = _resolve_schema_path_expr(
            schema_value,
            script_path=script_path,
            bundle_dir=bundle_dir,
            module_assignments=module_assignments,
            scope=scope,
            call_lineno=getattr(node, "lineno", 0),
            seen_names=set(),
        )
        if schema_path is None:
            raise ValueError(
                f"OpenClaw workflow step {step_id!r} must use a statically discoverable schema file "
                f"under prompts/ for run_json_batch in {script_path}"
            )
        _load_object_schema(schema_path, script_path=script_path, step_id=step_id)
        schema_paths.add(schema_path.resolve())

    if len(llm_agents) > 1:
        raise ValueError(
            f"Python step script {script_path} contains multiple run_json_batch agent values: {sorted(llm_agents)}"
        )
    resolved_batch_kinds = {kind for kind in batch_kinds if kind != "unknown"}
    if len(resolved_batch_kinds) > 1:
        raise ValueError(
            f"Python step script {script_path} mixes singleton and multi-item run_json_batch calls: "
            f"{sorted(resolved_batch_kinds)}"
        )
    if saw_nonliteral_agent and not llm_agents:
        raise ValueError(
            f"Python step script {script_path} uses run_json_batch without a literal agent=... value"
        )

    llm_agent = next(iter(llm_agents), None)
    if llm_agent is not None and llm_agent != step_id:
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} uses run_json_batch(agent={llm_agent!r}) in {script_path}. "
            "For Python-step OpenClaw workflows, the literal agent= value must exactly match the workflow step id."
        )
    if llm_agent is not None and not schema_paths:
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} must use a statically discoverable schema file in {script_path}"
        )
    if len(schema_paths) > 1:
        rendered = ", ".join(str(path) for path in sorted(schema_paths))
        raise ValueError(
            f"OpenClaw workflow step {step_id!r} references multiple output schemas in {script_path}: {rendered}"
        )

    return _StepScriptInfo(
        path=script_path,
        artifact_reads=tuple(sorted(artifact_reads)),
        llm_agent=llm_agent,
        batch_kind=next(iter(resolved_batch_kinds), "unknown") if llm_agent is not None else None,
        output_schema_path=next(iter(schema_paths), None),
    )


def _script_info_by_step(bundle_dir: Path, step_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, _StepScriptInfo]:
    infos: Dict[str, _StepScriptInfo] = {}
    for step_id, step in step_by_id.items():
        script_path = _resolve_python_script(bundle_dir, step.get("command"))
        if script_path is None:
            continue
        infos[step_id] = _inspect_python_script(step_id, script_path)
    return infos


def _is_llm_step(step_id: str, script_infos: Dict[str, _StepScriptInfo]) -> bool:
    info = script_infos.get(step_id)
    return info is not None and info.llm_agent is not None


def parse_lobster_workflow(path: str) -> Dict[str, Any]:
    workflow_path = Path(path)
    if not workflow_path.exists():
        raise FileNotFoundError(f"Lobster workflow file not found: {workflow_path}")

    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Invalid Lobster workflow (missing non-empty steps): {workflow_path}")

    ordered_step_ids: List[str] = []
    step_by_id: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            continue
        ordered_step_ids.append(step_id)
        step_by_id[step_id] = step

    if not ordered_step_ids:
        raise ValueError(f"No valid step IDs found in Lobster workflow: {workflow_path}")

    script_infos = _script_info_by_step(workflow_path.parent, step_by_id)

    deps: Dict[str, List[str]] = {}
    for step_id in ordered_step_ids:
        step = step_by_id[step_id]
        references = set()
        references.update(_extract_step_refs(step.get("stdin")))
        references.update(_extract_step_refs(step.get("command")))

        script_info = script_infos.get(step_id)
        if script_info is not None:
            references.update(script_info.artifact_reads)

        deps[step_id] = sorted(ref for ref in references if ref in step_by_id and ref != step_id)

    llm_step_ids = [step_id for step_id in ordered_step_ids if _is_llm_step(step_id, script_infos)]
    if not llm_step_ids:
        raise ValueError(
            f"No profiled LLM steps detected in Lobster workflow: {workflow_path}. "
            "Expected Python step scripts that call run_json_batch(..., agent='<step_id>', ...)."
        )
    llm_set = set(llm_step_ids)
    llm_pos = {step_id: idx for idx, step_id in enumerate(llm_step_ids)}

    upstream_cache: Dict[str, List[str]] = {}

    def upstream_llm(step_id: str) -> List[str]:
        cached = upstream_cache.get(step_id)
        if cached is not None:
            return cached
        stack = list(deps.get(step_id, []))
        seen: Set[str] = set()
        found: Set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in llm_set:
                found.add(cur)
                continue
            stack.extend(deps.get(cur, []))
        ordered = sorted(found, key=lambda item: llm_pos[item])
        upstream_cache[step_id] = ordered
        return ordered

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_edges: Set[tuple[str, str, str]] = set()

    for step_id in llm_step_ids:
        upstream = upstream_llm(step_id)
        batch_kind = (script_infos.get(step_id).batch_kind if script_infos.get(step_id) is not None else "unknown")
        if batch_kind == "single" and len(upstream) >= 2:
            operator = "map_reduce"
            inputs: Dict[str, Any] = {
                "items": [{"ref": f"state.{upstream_step}"} for upstream_step in upstream]
            }
            edge_sources = upstream
        elif batch_kind == "single" and len(upstream) == 1:
            operator = "reduce"
            inputs = {"items": [{"ref": f"state.{upstream[0]}"}]}
            edge_sources = upstream
        else:
            operator = "map"
            chosen = upstream[-1:]
            if chosen:
                inputs = {"source": {"ref": f"state.{chosen[0]}"}}
            else:
                inputs = {}
            edge_sources = chosen

        nodes.append(
            {
                "id": step_id,
                "type": "agent",
                "name": step_id,
                "llm_ref": step_id,
                "io": {"inputs": inputs},
                "metadata": {
                    "operator": operator,
                    "output_schema_path": str(script_infos[step_id].output_schema_path)
                    if script_infos.get(step_id) is not None and script_infos[step_id].output_schema_path is not None
                    else "",
                },
            }
        )

        for src in edge_sources:
            key = (src, step_id, operator)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"from": src, "to": step_id, "operator": operator})

    outputs_ref = llm_step_ids[-1]
    spec = {
        "version": "v1",
        "name": str(payload.get("name") or workflow_path.stem),
        "metadata": {
            "source": "lobster_yaml",
            "workflow_type": "openclaw_lobster",
            "workflow_file": str(workflow_path),
        },
        "nodes": nodes,
        "edges": edges,
        "entry": llm_step_ids[0],
        "outputs": {
            "final_answer": {"ref": f"state.{outputs_ref}"},
            "full_solution": {"ref": f"state.{outputs_ref}"},
            "final_solution": {"ref": f"state.{outputs_ref}"},
        },
    }
    return spec


__all__ = ["parse_lobster_workflow"]
