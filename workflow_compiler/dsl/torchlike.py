"""PyTorch-like Python DSL for FlowCompile workflows.

This module provides a capture-based compiler that turns normal Python
workflow definitions into a JSON workflow spec. It supports:
- Normal Python `for` loops over `range(...)` (unrolled at compile time)
- `if cond: break` inside those loops (captured as a break condition)
- No explicit DSL helpers in user code

Limitations (by design):
- Only `if <cond>: break` is recognized for break capture.
- The break condition should use a field from a node output, e.g.
  `if test_out["test_passed"]: break` or `if test_out.test_passed: break`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import ast
import builtins
import inspect
import textwrap
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from workflow_compiler.dsl.structure_inference import infer_structures


# =========================
# Condition Expressions
# =========================

@dataclass(frozen=True)
class ConditionExpr:
    path: str
    op: str = "truthy"
    value: Any = None

    def invert(self) -> "ConditionExpr":
        inverse = {
            "truthy": "falsy",
            "falsy": "truthy",
            "eq": "ne",
            "ne": "eq",
            "gt": "lte",
            "gte": "lt",
            "lt": "gte",
            "lte": "gt",
            "in": "not_in",
            "not_in": "in",
            "contains": "not_contains",
            "not_contains": "contains",
        }
        return ConditionExpr(path=self.path, op=inverse.get(self.op, self.op), value=self.value)

    def to_when(self) -> Dict[str, Any]:
        when = {"path": self.path, "op": self.op}
        if self.value is not None and self.op not in ("truthy", "falsy"):
            when["value"] = self.value
        return when


@dataclass(frozen=True)
class FieldRef:
    path: str

    def _cmp(self, op: str, value: Any) -> ConditionExpr:
        return ConditionExpr(path=self.path, op=op, value=value)

    def __eq__(self, other: Any) -> ConditionExpr:  # type: ignore[override]
        return self._cmp("eq", other)

    def __ne__(self, other: Any) -> ConditionExpr:  # type: ignore[override]
        return self._cmp("ne", other)

    def __gt__(self, other: Any) -> ConditionExpr:
        return self._cmp("gt", other)

    def __ge__(self, other: Any) -> ConditionExpr:
        return self._cmp("gte", other)

    def __lt__(self, other: Any) -> ConditionExpr:
        return self._cmp("lt", other)

    def __le__(self, other: Any) -> ConditionExpr:
        return self._cmp("lte", other)

    def __bool__(self) -> bool:
        # During capture, always allow `if field:` to evaluate as True
        # so that `if cond: break` calls our break hook.
        ctx = CaptureContext.current()
        if ctx is not None and ctx.capturing:
            return True
        # Outside capture, best effort: treat as truthy
        return True


@dataclass(frozen=True)
class NodeOutput:
    node_id: str

    def field(self, name: str) -> FieldRef:
        return FieldRef(path=f"state.{self.node_id}.{name}")

    def __getitem__(self, key: str) -> FieldRef:
        return self.field(key)

    def __getattr__(self, name: str) -> FieldRef:
        if name.startswith("__"):
            raise AttributeError(name)
        return self.field(name)

    def get(self, key: str, default: Any = None) -> FieldRef:  # type: ignore[override]
        return self.field(key)

    def ref(self) -> str:
        return f"state.{self.node_id}"

    def __bool__(self) -> bool:
        ctx = CaptureContext.current()
        if ctx is not None and ctx.capturing:
            return True
        return True


@dataclass(frozen=True)
class InputRef:
    name: str

    def ref(self) -> str:
        return f"input.{self.name}"

    def field(self, name: str) -> FieldRef:
        return FieldRef(path=f"input.{self.name}.{name}")

    def __getitem__(self, key: str) -> FieldRef:
        return self.field(key)

    def __getattr__(self, name: str) -> FieldRef:
        if name.startswith("__"):
            raise AttributeError(name)
        return self.field(name)


# =========================
# DSL Nodes
# =========================

class Node:
    def __init__(
        self,
        name: str,
        node_type: str,
        llm: Optional[str] = None,
        prompt: Optional[str] = None,
        prompt_ref: Optional[str] = None,
        impl: Optional[str] = None,
    ) -> None:
        self.name = name
        self.node_type = node_type
        self.llm_ref = llm
        self.prompt = prompt
        self.prompt_ref = prompt_ref
        self.impl = impl

    def __call__(self, **kwargs) -> NodeOutput:
        ctx = CaptureContext.current()
        if ctx is None or not ctx.capturing:
            raise RuntimeError("Node calls are only supported during DSL capture/compile")
        return ctx.record_call(self, kwargs)


class AgentNode(Node):
    def __init__(self, name: str, llm: Optional[str] = None, prompt: Optional[str] = None, prompt_ref: Optional[str] = None):
        super().__init__(name=name, node_type="agent", llm=llm, prompt=prompt, prompt_ref=prompt_ref)


class ToolNode(Node):
    def __init__(self, name: str, impl: str):
        super().__init__(name=name, node_type="tool", impl=impl)


# =========================
# Capture Context
# =========================

@dataclass
class CallRecord:
    call_id: str
    node: Node
    inputs: Dict[str, Any]


@dataclass
class BreakEdge:
    source_id: str
    cond: ConditionExpr


@dataclass
class LoopContext:
    loop_id: str
    break_exits: List[BreakEdge]


class CaptureContext:
    _CURRENT: "CaptureContext" | None = None

    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.capturing = False
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.calls: List[CallRecord] = []
        self.edges: List[Dict[str, Any]] = []
        self.last_call_id: Optional[str] = None
        self.call_counts: Dict[str, int] = {}
        self.loop_stack: List[LoopContext] = []
        self.pending_not_cond: Optional[BreakEdge] = None
        self.pending_loop_exits: List[BreakEdge] = []
        self.after_loop = False

    @classmethod
    def current(cls) -> Optional["CaptureContext"]:
        return cls._CURRENT

    @contextmanager
    def capture(self):
        prev = CaptureContext._CURRENT
        CaptureContext._CURRENT = self
        self.capturing = True
        try:
            yield self
        finally:
            self.capturing = False
            CaptureContext._CURRENT = prev

    def _next_call_id(self, base_name: str) -> str:
        count = self.call_counts.get(base_name, 0) + 1
        self.call_counts[base_name] = count
        if count == 1:
            return base_name
        return f"{base_name}_{count}"

    def record_call(self, node: Node, inputs: Dict[str, Any]) -> NodeOutput:
        call_id = self._next_call_id(node.name)
        call = CallRecord(call_id=call_id, node=node, inputs=inputs)
        self.calls.append(call)

        if call_id not in self.nodes:
            self.nodes[call_id] = {
                "id": call_id,
                "type": node.node_type,
                "name": node.name,
                "llm_ref": node.llm_ref,
                "prompt": node.prompt,
                "prompt_ref": node.prompt_ref,
                "impl": node.impl,
                "io": {"inputs": self._serialize_inputs(inputs)},
            }

        if self.last_call_id is not None:
            edge: Dict[str, Any] = {"from": self.last_call_id, "to": call_id}
            if self.pending_not_cond and self.pending_not_cond.source_id == self.last_call_id:
                edge["when"] = self.pending_not_cond.cond.invert().to_when()
                self.pending_not_cond = None
            self.edges.append(edge)

        if self.after_loop and self.pending_loop_exits:
            for br in self.pending_loop_exits:
                self.edges.append({"from": br.source_id, "to": call_id, "when": br.cond.to_when()})
            self.pending_loop_exits = []
            self.after_loop = False

        self.last_call_id = call_id
        return NodeOutput(node_id=call_id)

    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        def serialize_value(val: Any) -> Any:
            if isinstance(val, NodeOutput):
                return {"ref": val.ref()}
            if isinstance(val, InputRef):
                return {"ref": val.ref()}
            if isinstance(val, FieldRef):
                return {"ref": val.path}
            if isinstance(val, list):
                return [serialize_value(v) for v in val]
            if isinstance(val, dict):
                return {k: serialize_value(v) for k, v in val.items()}
            return val
        return {k: serialize_value(v) for k, v in inputs.items()}

    def enter_loop(self, loop_id: str) -> None:
        self.loop_stack.append(LoopContext(loop_id=loop_id, break_exits=[]))

    def exit_loop(self) -> None:
        if not self.loop_stack:
            return
        loop = self.loop_stack.pop()
        if loop.break_exits:
            self.pending_loop_exits.extend(loop.break_exits)
            self.after_loop = True

    def register_break(self, cond: ConditionExpr) -> None:
        if not self.loop_stack:
            return
        if self.last_call_id is None:
            return
        br = BreakEdge(source_id=self.last_call_id, cond=cond)
        self.loop_stack[-1].break_exits.append(br)
        self.pending_not_cond = br

    def finalize(self) -> None:
        if self.pending_loop_exits:
            end_id = "__end__"
            if end_id not in self.nodes:
                self.nodes[end_id] = {
                    "id": end_id,
                    "type": "end",
                    "name": "end",
                }
            for br in self.pending_loop_exits:
                self.edges.append({"from": br.source_id, "to": end_id, "when": br.cond.to_when()})
            self.pending_loop_exits = []
            self.after_loop = False


# =========================
# Builtins helpers for capture
# =========================

class DslRange:
    def __init__(self, args, ctx: CaptureContext, orig_range):
        self.args = args
        self.ctx = ctx
        self._orig_range = orig_range
        self.loop_id = f"loop_{id(self)}"

    def __iter__(self):
        self.ctx.enter_loop(self.loop_id)
        try:
            for i in self._orig_range(*self.args):
                yield i
        finally:
            self.ctx.exit_loop()


# =========================
# AST Transform
# =========================

class BreakTransformer(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> ast.AST:
        if len(node.body) == 1 and isinstance(node.body[0], ast.Break):
            # Replace: if cond: break  -> if cond: __dsl_break(cond)
            new_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__dsl_break", ctx=ast.Load()),
                    args=[node.test],
                    keywords=[],
                )
            )
            node.body = [new_call]
            return node
        # For other ifs, continue recursion
        return self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> ast.AST:
        # Any raw break that wasn't handled by an If is unsupported
        raise SyntaxError("Unsupported break usage. Use 'if <cond>: break'.")


# =========================
# Workflow Module
# =========================

class MetricContext:
    """Vectorized metric helper for workflow backward formulas."""

    REQUIRED_COLUMNS = ("setting", "accuracy", "latency")

    def __init__(self, workflow: "WorkflowModule", payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        del metadata
        payload = workflow._validate_backward_payload(payload)
        self.workflow = workflow
        self.payload = payload
        self.structure: Dict[str, Any] = payload["structure"]
        self.metrics: Dict[str, pd.DataFrame] = payload.get("metrics") or {}

        self.metric_agents: List[str] = workflow.infer_agent_names()
        raw_counts = self.structure.get("active_agent_counts") or {}
        self.active_agent_counts: Dict[str, int] = {
            str(agent): int(raw_counts.get(agent, 0))
            for agent in raw_counts.keys()
        }
        for agent in self.metric_agents:
            self.active_agent_counts.setdefault(agent, int(raw_counts.get(agent, 0)))

        self._active_metric_agents: List[str] = []
        self._metric_frames: Dict[str, pd.DataFrame] = {}
        for agent in self.metric_agents:
            if not self.enabled(agent):
                continue
            df = self.metrics.get(agent)
            if df is None:
                # Missing tables are allowed; backward formulas can use defaults.
                continue
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Metrics DataFrame for agent '{agent}' is missing columns: {', '.join(missing_cols)}"
                )
            trimmed = df[list(self.REQUIRED_COLUMNS)].reset_index(drop=True)
            if trimmed.empty:
                raise ValueError(f"Metrics DataFrame for active agent '{agent}' is empty.")
            self._active_metric_agents.append(agent)
            self._metric_frames[agent] = trimmed

        self._shape: Tuple[int, ...] = ()
        self._meshgrids: List[np.ndarray] = []
        self._acc_arrays: Dict[str, Any] = {}
        self._lat_arrays: Dict[str, Any] = {}
        self._setting_arrays: Dict[str, Any] = {}

        if self._active_metric_agents:
            shape = []
            for agent in self._active_metric_agents:
                shape.append(len(self._metric_frames[agent]))
            self._shape = tuple(shape)

            indices = [np.arange(size) for size in self._shape]
            self._meshgrids = list(np.meshgrid(*indices, indexing="ij"))

            n_dims = len(self._active_metric_agents)
            for idx, agent in enumerate(self._active_metric_agents):
                frame = self._metric_frames[agent]
                reshape_dims = [1] * n_dims
                reshape_dims[idx] = len(frame)
                self._acc_arrays[agent] = frame["accuracy"].to_numpy().reshape(reshape_dims)
                self._lat_arrays[agent] = frame["latency"].to_numpy().reshape(reshape_dims)
                self._setting_arrays[agent] = frame["setting"].to_numpy()
        else:
            self._shape = (1,)

    @property
    def active_metric_agents(self) -> List[str]:
        return list(self._active_metric_agents)

    @property
    def n_rows(self) -> int:
        if not self._shape:
            return 1
        return int(np.prod(self._shape))

    def count(self, agent: str) -> int:
        return int(self.active_agent_counts.get(agent, 0))

    def enabled(self, agent: str) -> bool:
        return self.count(agent) > 0

    def acc(self, agent: str, default: Any = 0.0) -> Any:
        if not self.enabled(agent):
            return default
        if agent not in self._acc_arrays:
            return default
        return self._acc_arrays[agent]

    def lat(self, agent: str, default: Any = 0.0) -> Any:
        if not self.enabled(agent):
            return default
        if agent not in self._lat_arrays:
            return default
        return self._lat_arrays[agent]

    @staticmethod
    def _broadcast(value: Any, rows: int) -> List[Any]:
        if isinstance(value, pd.Series):
            value = value.to_numpy()
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            if len(flat) == rows:
                return flat.tolist()
            if len(flat) == 1:
                return [flat.item()] * rows
        if isinstance(value, (list, tuple)):
            if len(value) == rows:
                return list(value)
            if len(value) == 1:
                return [value[0]] * rows
        return [value] * rows

    def finish(
        self,
        workflow_accuracy: Any,
        workflow_latency: Any,
        extra_cols: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        shape = self._shape if self._shape else (1,)
        accuracy = np.broadcast_to(np.asarray(workflow_accuracy), shape).reshape(-1)
        latency = np.broadcast_to(np.asarray(workflow_latency), shape).reshape(-1)
        rows = int(len(accuracy))

        structure_id = str(self.structure.get("structure_id", ""))
        total_branches = int(self.structure.get("total_branches", 0))
        is_full = bool(self.structure.get("is_full", False))

        result: Dict[str, Any] = {
            "workflow_accuracy": accuracy,
            "workflow_latency": latency,
            "structure_id": [structure_id] * rows,
            "total_branches": [total_branches] * rows,
            "is_full": [is_full] * rows,
        }

        for agent, count in sorted(self.active_agent_counts.items()):
            result[f"{agent}_count"] = [int(count)] * rows

        agent_index = {agent: idx for idx, agent in enumerate(self._active_metric_agents)}
        for agent in self.metric_agents:
            col = f"{agent}_setting"
            idx = agent_index.get(agent)
            if idx is None:
                result[col] = [None] * rows
                continue
            settings = self._setting_arrays[agent][self._meshgrids[idx].reshape(-1)]
            result[col] = settings

        for key, value in (extra_cols or {}).items():
            result[key] = self._broadcast(value, rows)

        return pd.DataFrame(result)


class WorkflowModule:
    workflow_type = "unknown"

    def __init__(self, name: Optional[str] = None, execution_mode: str = "sequential"):
        from workflow_compiler.dsl.auto_backward import validate_execution_mode

        self.name = name or self.__class__.__name__
        self.execution_mode = validate_execution_mode(execution_mode)
        self._structures_cache: Optional[List[Dict[str, Any]]] = None
        self._compiled_spec_cache: Optional[Dict[str, Any]] = None

    def forward(self, *args, **kwargs):  # pragma: no cover - user-defined
        raise NotImplementedError

    def backward(self, payload: Dict[str, Any]) -> pd.DataFrame:
        from workflow_compiler.dsl.auto_backward import auto_backward

        return auto_backward(self, payload)

    @staticmethod
    def _validate_backward_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("Backward payload must be a dict.")
        structure = payload.get("structure")
        if not isinstance(structure, dict):
            raise ValueError("Backward payload must include a dict 'structure'.")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Backward payload must include dict 'metrics'.")
        return payload

    def metric_context(self, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> MetricContext:
        if metadata is None and isinstance(payload, dict):
            candidate = payload.get("metadata")
            if isinstance(candidate, dict):
                metadata = candidate
        return MetricContext(self, payload, metadata=metadata)

    def _compile_cached(self) -> Dict[str, Any]:
        if self._compiled_spec_cache is None:
            self._compiled_spec_cache = self.compile()
        return self._compiled_spec_cache

    @staticmethod
    def _validate_canonical_llm_refs(spec: Dict[str, Any]) -> None:
        mismatched: List[str] = []
        for node in spec.get("nodes", []) or []:
            if node.get("type") != "agent":
                continue
            name = str(node.get("name") or "")
            llm_ref = node.get("llm_ref")
            if llm_ref is None:
                continue
            if str(llm_ref) != name:
                mismatched.append(f"{name}->{llm_ref}")
        if mismatched:
            raise ValueError(
                "Canonical naming requires AgentNode.name == llm key. "
                f"Found non-canonical llm_ref override(s): {', '.join(mismatched)}"
            )

    def infer_agent_names(self) -> List[str]:
        spec = self._compile_cached()
        self._validate_canonical_llm_refs(spec)
        names: List[str] = []
        seen: set = set()
        for node in spec.get("nodes", []) or []:
            if node.get("type") != "agent":
                continue
            name = str(node.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def infer_metric_agents(self) -> List[str]:
        return self.infer_agent_names()

    def infer_profiling_agents(self) -> List[str]:
        return self.infer_agent_names()

    @staticmethod
    def _count_agent_settings(
        df_subagents: Dict[str, pd.DataFrame],
        agents: List[str],
    ) -> int:
        count = 1
        for agent in agents:
            df = df_subagents.get(agent)
            if df is None:
                return 0
            n_settings = len(df)
            if n_settings <= 0:
                return 0
            count *= n_settings
        return count

    def _estimate_structure_config_count(
        self,
        structure: Dict[str, Any],
        df_subagents: Dict[str, pd.DataFrame],
        required_agents: List[str],
    ) -> int:
        active_counts = structure.get("active_agent_counts") or {}
        active_metric_agents = [
            agent for agent in required_agents if int(active_counts.get(agent, 0)) > 0
        ]
        if not active_metric_agents:
            return 1
        return self._count_agent_settings(df_subagents, active_metric_agents)

    def normalize_subagent_name(self, name: str) -> str:
        return name

    def normalize_subagent_stats(self, df_subagents: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {str(key): value.copy() for key, value in df_subagents.items()}

    def runtime_key_for_agent(self, agent: str) -> str:
        return agent

    def enumerate_structures(self) -> List[Dict[str, Any]]:
        if self._structures_cache is None:
            inferred = infer_structures(spec=self._compile_cached())
            self._structures_cache = inferred

        return [copy.deepcopy(structure) for structure in self._structures_cache]

    def get_full_structure(self) -> Dict[str, Any]:
        structures = list(self.enumerate_structures())
        if not structures:
            raise ValueError(f"No inferred structures available for workflow '{self.workflow_type}'")
        for structure in structures:
            if bool(structure.get("is_full", False)):
                return structure
        return max(
            structures,
            key=lambda item: (
                int(item.get("total_branches", 0)),
                len(item.get("active_node_ids") or []),
            ),
        )

    def get_structure(self, structure_id: str) -> Dict[str, Any]:
        for structure in self.enumerate_structures():
            if structure.get("structure_id") == structure_id:
                return structure
        raise ValueError(f"Unknown structure_id '{structure_id}' for workflow '{self.workflow_type}'")

    def _extract_agent_setting(self, config: Dict[str, Any], agent: str) -> Optional[str]:
        agents = config.get("agents") or {}
        agent_info = agents.get(agent) or {}
        setting = agent_info.get("setting")
        if setting:
            return setting
        model = agent_info.get("model")
        budget = agent_info.get("budget")
        if model:
            if budget is None:
                return str(model)
            return f"{model}_budget_{budget}"
        return None

    @staticmethod
    def _single_agent_row(source_df: pd.DataFrame, setting: str, agent: str) -> pd.DataFrame:
        row = source_df[source_df["setting"] == setting]
        if row.empty:
            raise ValueError(f"Setting '{setting}' not found for agent '{agent}'")
        return row[["setting", "accuracy", "latency"]].iloc[[0]].reset_index(drop=True)

    def compute_configs(
        self,
        df_subagents: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        from workflow_compiler.core.analysis.prediction import (
            SearchSpaceSpec,
            apply_search_space_to_subagents,
            apply_structure_constraints,
        )

        required_agents = self.infer_metric_agents()
        search_spec = SearchSpaceSpec.from_dict((metadata or {}).get("search_space"))

        filtered_subagents, filter_info = apply_search_space_to_subagents(
            df_subagents,
            required_agents=required_agents,
            spec=search_spec,
        )

        all_structures = list(self.enumerate_structures())
        structures, structure_info = apply_structure_constraints(all_structures, search_spec)
        show_progress = bool((metadata or {}).get("show_progress", True))
        progress_desc = (metadata or {}).get("progress_desc", f"Enumerating {self.workflow_type} configs")

        dfs: List[pd.DataFrame] = []
        structure_estimates: List[int] = [
            self._estimate_structure_config_count(structure, filtered_subagents, required_agents)
            for structure in structures
        ]

        total_estimated_configs: Optional[int] = int(sum(structure_estimates)) if structure_estimates else None

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=total_estimated_configs,
                desc=progress_desc,
                unit="cfg",
                leave=False,
            )

        for structure in structures:
            metrics_payload = {
                agent: filtered_subagents[agent]
                for agent in required_agents
                if agent in filtered_subagents
            }
            df = self.backward(
                {
                    "structure": structure,
                    "metrics": metrics_payload,
                    "metadata": metadata or {},
                }
            )
            produced = 0 if df is None else len(df)
            if progress_bar is not None and produced > 0:
                if progress_bar.total is not None and progress_bar.n + produced > progress_bar.total:
                    progress_bar.total = progress_bar.n + produced
                    progress_bar.refresh()
                progress_bar.update(produced)

            if produced > 0:
                dfs.append(df)

        if progress_bar is not None:
            if progress_bar.total is not None and progress_bar.n < progress_bar.total:
                progress_bar.total = progress_bar.n
                progress_bar.refresh()
            progress_bar.close()

        if not dfs:
            return pd.DataFrame()

        merged = pd.concat(dfs, ignore_index=True)
        merged.attrs["search_space_resolved"] = {
            **filter_info,
            **structure_info,
            "search_axes": sorted(search_spec.search_axes),
        }
        return merged

    def estimate_metrics(
        self,
        config: Dict[str, Any],
        df_subagents: Dict[str, pd.DataFrame],
    ) -> Tuple[float, float]:
        structure_id = config.get("structure_id")
        if not structure_id:
            raise ValueError("Missing structure_id in config")
        structure = self.get_structure(structure_id)
        active_counts = structure.get("active_agent_counts") or {}

        metrics_payload: Dict[str, pd.DataFrame] = {}
        for agent in self.infer_agent_names():
            if int(active_counts.get(agent, 0)) <= 0:
                continue
            setting = self._extract_agent_setting(config, agent)
            if setting is None:
                raise ValueError(f"Missing setting for active agent '{agent}'")
            if agent not in df_subagents:
                raise ValueError(f"Missing subagent data for '{agent}'")
            metrics_payload[agent] = self._single_agent_row(df_subagents[agent], setting, agent)

        df = self.backward({"structure": structure, "metrics": metrics_payload})
        if df is None or df.empty:
            return 0.0, 0.0
        return float(df["workflow_accuracy"].iloc[0]), float(df["workflow_latency"].iloc[0])

    def _build_capture_inputs(self) -> Dict[str, InputRef]:
        sig = inspect.signature(self.forward)
        inputs: Dict[str, InputRef] = {}
        for name, _param in sig.parameters.items():
            if name == "self":
                continue
            inputs[name] = InputRef(name=name)
        return inputs

    def _transform_forward(self, ctx: CaptureContext):
        src = textwrap.dedent(inspect.getsource(self.forward))
        tree = ast.parse(src)
        transformer = BreakTransformer()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        compiled = compile(tree, filename=inspect.getsourcefile(self.forward) or "<dsl>", mode="exec")
        ns: Dict[str, Any] = {}
        glb = dict(self.forward.__globals__)
        glb["__dsl_break"] = ctx_break_hook(ctx)
        exec(compiled, glb, ns)
        return ns[self.forward.__name__]

    def compile(self) -> Dict[str, Any]:
        ctx = CaptureContext(self.name)
        inputs = self._build_capture_inputs()

        transformed_forward = self._transform_forward(ctx)

        with ctx.capture():
            orig_range = builtins.range

            def dsl_range(*args):
                return DslRange(args, ctx, orig_range)

            builtins.range = dsl_range  # type: ignore
            try:
                result = transformed_forward(self, **inputs)
            finally:
                builtins.range = orig_range  # type: ignore

        ctx.finalize()
        output_spec = serialize_output(result)

        metadata = {"source": "python_dsl"}
        if hasattr(self, "workflow_type"):
            metadata["workflow_type"] = getattr(self, "workflow_type")
        spec = {
            "version": "v1",
            "name": self.name,
            "metadata": metadata,
            "nodes": list(ctx.nodes.values()),
            "edges": ctx.edges,
            "entry": ctx.calls[0].call_id if ctx.calls else None,
            "outputs": output_spec,
        }
        return spec

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.compile(), indent=indent)


# =========================
# Helpers
# =========================

def ctx_break_hook(ctx: CaptureContext):
    def _hook(cond: Any):
        condition = normalize_condition(cond)
        if condition is not None:
            ctx.register_break(condition)
        return False
    return _hook


def normalize_condition(cond: Any) -> Optional[ConditionExpr]:
    if isinstance(cond, ConditionExpr):
        return cond
    if isinstance(cond, FieldRef):
        return ConditionExpr(path=cond.path, op="truthy")
    return None


def serialize_output(value: Any) -> Any:
    if isinstance(value, NodeOutput):
        return {"ref": value.ref()}
    if isinstance(value, InputRef):
        return {"ref": value.ref()}
    if isinstance(value, FieldRef):
        return {"ref": value.path}
    if isinstance(value, list):
        return [serialize_output(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_output(v) for k, v in value.items()}
    return value


__all__ = [
    "WorkflowModule",
    "MetricContext",
    "AgentNode",
    "ToolNode",
]
