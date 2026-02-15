"""JSON workflow executor for DSL specs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from workflow_compiler.core.llm.client import create_llm_instance
from workflow_compiler.core.llm.config import parse_config, ThinkingBudgetLLM
from workflow_compiler.core.llm.config import build_setting
from workflow_compiler.dsl.registry import get_agent_factory, get_tool


def _get_path(data: Any, path: str) -> Any:
    parts = path.split(".")
    cur: Any = data
    for part in parts:
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            break
    return cur


def _eval_when(when: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if not when:
        return True
    if "all" in when:
        return all(_eval_when(w, state) for w in when.get("all") or [])
    if "any" in when:
        return any(_eval_when(w, state) for w in when.get("any") or [])

    path = when.get("path")
    op = when.get("op", "truthy")
    value = when.get("value")
    actual = _get_path({"state": state}, path) if path else None

    if op == "truthy":
        return bool(actual)
    if op == "falsy":
        return not bool(actual)
    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "gt":
        return actual > value
    if op == "gte":
        return actual >= value
    if op == "lt":
        return actual < value
    if op == "lte":
        return actual <= value
    if op == "in":
        return actual in value if value is not None else False
    if op == "not_in":
        return actual not in value if value is not None else True
    if op == "contains":
        return value in actual if actual is not None else False
    if op == "len_gt":
        return len(actual) > value if actual is not None else False
    if op == "len_gte":
        return len(actual) >= value if actual is not None else False
    if op == "len_lt":
        return len(actual) < value if actual is not None else False
    if op == "len_lte":
        return len(actual) <= value if actual is not None else False
    if op == "len_eq":
        return len(actual) == value if actual is not None else False
    return False


def _resolve_refs(value: Any, inputs: Dict[str, Any], state: Dict[str, Any]) -> Any:
    if isinstance(value, dict) and "ref" in value:
        ref = value.get("ref")
        if isinstance(ref, str):
            if ref.startswith("state."):
                return _get_path({"state": state}, ref)
            if ref.startswith("input."):
                return _get_path({"input": inputs}, ref)
        return value
    if isinstance(value, list):
        return [_resolve_refs(v, inputs, state) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, inputs, state) for k, v in value.items()}
    return value


class DslExecutor:
    def __init__(self, spec: Dict[str, Any], workflow_type: str, config: Dict[str, Any]):
        self.spec = spec
        self.workflow_type = workflow_type
        self.config = config or {}
        self.nodes = {n.get("id"): n for n in spec.get("nodes", [])}
        self.edges = spec.get("edges", [])
        self.entry = spec.get("entry")
        self._agent_instances: Dict[str, Any] = {}

    def _build_llm(self, llm_ref: str):
        agents_cfg = self.config.get("agents") or {}
        agent_info = agents_cfg.get(llm_ref, {})
        setting = agent_info.get("setting")
        if not setting:
            setting = build_setting(agent_info.get("model"), agent_info.get("budget"))
        model, budget = parse_config(setting) if setting else (None, None)
        if model is None:
            available_refs = sorted(k for k in agents_cfg.keys() if k)
            available_msg = ", ".join(available_refs) if available_refs else "<none>"
            raise ValueError(
                f"Missing LLM setting for active operator '{llm_ref}'. "
                f"Available configured operators: {available_msg}."
            )
        base_llm = create_llm_instance(model)
        return ThinkingBudgetLLM(base_llm, budget)

    def _get_agent(self, node: Dict[str, Any]):
        name = node.get("name")
        if name in self._agent_instances:
            return self._agent_instances[name]
        factory = get_agent_factory(self.workflow_type, name)
        if factory is None:
            raise ValueError(f"No agent factory for {self.workflow_type}:{name}")
        llm_ref = node.get("llm_ref") or name
        llm = self._build_llm(llm_ref)
        agent = factory(llm)
        self._agent_instances[name] = agent
        return agent

    async def aclose(self) -> None:
        """Close all cached agent LLM clients."""
        for agent in list(self._agent_instances.values()):
            llm = getattr(agent, "llm", None)
            close_method = getattr(llm, "aclose", None) if llm is not None else None
            if close_method:
                try:
                    await close_method()
                except Exception:
                    pass
        self._agent_instances.clear()

    async def run(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        state: Dict[str, Any] = {}
        steps: List[Dict[str, Any]] = []

        outgoing: Dict[str, List[Dict[str, Any]]] = {}
        for e in self.edges:
            outgoing.setdefault(e.get("from"), []).append(e)

        node_id = self.entry
        step_num = 0
        while node_id:
            node = self.nodes.get(node_id)
            if node is None:
                break
            step_num += 1
            node_type = node.get("type")
            io = node.get("io") or {}
            input_spec = (io.get("inputs") if isinstance(io, dict) else {}) or {}
            call_kwargs = _resolve_refs(input_spec, inputs, state)

            if node_type == "agent":
                agent = self._get_agent(node)
                output, metadata = await agent.execute(**call_kwargs)
                state[node_id] = output
                steps.append({
                    "step": step_num,
                    "agent": node.get("name"),
                    "input": call_kwargs,
                    "output": output,
                    "metadata": metadata,
                })
            elif node_type == "tool":
                impl = node.get("impl")
                tool = get_tool(impl) if impl else None
                if tool is None:
                    raise ValueError(f"Unknown tool impl: {impl}")
                output = tool(**call_kwargs)
                state[node_id] = output
                steps.append({
                    "step": step_num,
                    "agent": node.get("name"),
                    "input": call_kwargs,
                    "output": output,
                    "metadata": {"type": "rule_based", "tool": impl},
                })
            else:
                state[node_id] = None
                steps.append({
                    "step": step_num,
                    "agent": node.get("name"),
                    "input": call_kwargs,
                    "output": None,
                    "metadata": {"type": "noop"},
                })

            # Choose next node based on edges and conditions
            next_id = None
            for e in outgoing.get(node_id, []):
                when = e.get("when")
                if when is None or _eval_when(when, state):
                    next_id = e.get("to")
                    break
            node_id = next_id

        outputs = _resolve_refs(self.spec.get("outputs") or {}, inputs, state)
        return outputs, steps, state
