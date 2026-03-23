"""FlashFlow CLI."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

def _coerce_vllm_value(value: Optional[str]) -> Any:
    if value is None:
        return True
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    return value


def parse_vllm_flags(argv: List[str]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if not token.startswith("--"):
            raise SystemExit(f"Unsupported positional vLLM argument: {token}")
        key = token[2:].replace("-", "_")
        value: Optional[str] = None
        if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
            value = argv[idx + 1]
            idx += 2
        else:
            idx += 1
        flags[key] = _coerce_vllm_value(value)
    return flags


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="flashflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("workflow_dag_file")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args, extra = parser.parse_known_args(argv)
    if args.command != "serve":
        raise SystemExit(f"Unknown flashflow command '{args.command}'")

    from flashflow.runtime import FlashFlowRuntime

    runtime = FlashFlowRuntime(
        args.workflow_dag_file,
        vllm_args=parse_vllm_flags(extra),
    )
    from flashflow.server import create_app
    import uvicorn

    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
