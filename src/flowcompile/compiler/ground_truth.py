# -*- coding: utf-8 -*-
# @Date    : 2025-11-02
# @Desc    : Multi-agent workflow evaluation on MATH dataset
import warnings
warnings.filterwarnings("ignore")
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

from flowcompile.benchmarks.math import MATHBenchmark
from flowcompile.benchmarks.hotpotqa import HotpotQABenchmark
from flowcompile.benchmarks.livecodebench import LiveCodeBench
from flowcompile.core.logs import logger
from flowcompile.compiler.validation import (
    annotate_trace_file_with_scores,
    _metric_for_dataset,
    _score_is_success,
    _result_score,
)
from flowcompile.core.data_paths import resolve_required_path

# Import workflow classes and utilities from flowcompile
from flowcompile.dsl.runtime import DslWorkflowRunner
from flowcompile.core.llm.config import parse_config


async def _run_benchmark_with_annotation(
    benchmark,
    workflow,
    dataset: str,
    max_concurrent_tasks: int = 16,
):
    data = await benchmark.load_data()
    results = await benchmark.evaluate_all_problems(data, workflow, max_concurrent_tasks=max_concurrent_tasks)
    columns = benchmark.get_result_columns()
    average_score = benchmark.save_results_to_csv(results, columns)

    # Annotate trace with unified score/metric fields
    annotate_trace_file_with_scores(workflow.trace_file, dataset, results)

    metric_name = _metric_for_dataset(dataset)
    scores = [_result_score(dataset, r) for r in results]
    success_count = sum(1 for s in scores if _score_is_success(metric_name, float(s)))
    total_problems = len(scores)
    success_rate = success_count / total_problems if total_problems > 0 else 0.0

    return average_score, success_count, total_problems, success_rate

async def run_ground_truth(args):
    """
    Run DSL workflows for MATH/MATH500/GSM8K, HotpotQA, or LiveCodeBench.

    The CLI is responsible for constructing `args` with the same fields as the
    original step1 script.
    """
    
    profile_root = Path("results") / args.experiment_id / "01_profile"

    if args.task == "math" or args.task == "math500" or args.task == "gsm8k":
        # Set LLM configurations, using specific args if provided, otherwise fall back to --llm
        llm_configs = {
            'meta': args.meta_llm or args.llm,
            'programmer': args.programmer_llm or args.llm,
            'sc_ensemble': args.sc_ensemble_llm or args.llm,
            'refine_solver': args.refine_solver_llm or args.solver_llm or args.llm,
            'detailed_solver': args.detailed_solver_llm or args.solver_llm or args.llm,
            'generate_solver': args.generate_solver_llm or args.solver_llm or args.llm,
        }

        # Determine dataset name based on task
        if args.task == "math":
            dataset_name = "MATH"
        elif args.task == "gsm8k":
            dataset_name = "GSM8K"
        else:
            dataset_name = "MATH500"
        logger.info(f"Starting DSL Workflow evaluation on {dataset_name} validation set")
        logger.info(f"Raw LLM configurations: {llm_configs}")
        logger.info("Parsed configurations:")
        for agent_name, config in llm_configs.items():
            model, budget = parse_config(config)
            logger.info(f"  {agent_name}: model={model}, budget={budget}")

        # Default file paths differ between math, math500, and gsm8k
        if args.task == "math":
            file_path = resolve_required_path(args.file_path or "data/math_validate.jsonl", label="math validate file")
        elif args.task == "gsm8k":
            file_path = resolve_required_path(args.file_path or "data/gsm8k_validate.jsonl", label="gsm8k validate file")
        else:  # math500
            file_path = resolve_required_path(args.file_path or "data/math500_validate.jsonl", label="math500 validate file")
        if args.debug:
            output_dir = profile_root / "debug_dsl_agent"
            logger.info("Debug mode: Using DSL debug output directory")
            timestamp = "debug"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = profile_root / f"math_dsl_agent_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        workflow_type = "gsm8k" if args.task == "gsm8k" else "math"
        workflow = DslWorkflowRunner(
            name=f"dsl_math_solver{'_debug' if args.debug else ''}",
            llm_configs=llm_configs,
            workflow_type=workflow_type,
            output_dir=output_dir
        )

        math_benchmark = MATHBenchmark(
            name=dataset_name,  # Use dataset_name (MATH or MATH500)
            file_path=file_path,
            log_path=str(output_dir)
        )

        Path(math_benchmark.log_path).mkdir(parents=True, exist_ok=True)

        logger.info("Starting evaluation...")
        average_score, success_count, total_problems, success_rate = await _run_benchmark_with_annotation(
            math_benchmark,
            workflow,
            dataset_name,
            max_concurrent_tasks=16 if not args.debug else 1,
        )

        print("\n" + "="*80)
        print("DSL WORKFLOW EVALUATION COMPLETE")
        print("="*80)
        print(f"Experiment ID: {args.experiment_id}")
        print(f"Dataset: {dataset_name} Validation Set")
        print("Framework: Python DSL workflow")
        print("Workflow: Programmer → Refine → Detailed → Generate×2 → sc_ensemble")

        print("\nAgent Configurations:")
        for agent_name, model_config in llm_configs.items():
            model, budget = parse_config(model_config)
            budget_str = f"budget={budget}" if budget is not None else "no budget"
            print(f"  - {agent_name.capitalize()}: {model} ({budget_str})")
        print("\nResults:")
        print(f"  Accuracy: {average_score:.4f} ({average_score*100:.2f}%)")
        print(f"  Success Rate (Verification): {success_rate:.4f} ({success_rate*100:.2f}%)")
        print(f"  Total Problems: {total_problems}")
        print(f"  Verified Successful: {success_count}")
        print("\nOutput Files:")
        print(f"  Trace file: {workflow.trace_file}")
        print(f"  Results directory: {output_dir}")
        print("="*80)

        summary = {
            "workflow": f"dsl_math_solver{'_debug' if args.debug else ''}",
            "workflow_type": "dsl",
            "dataset": f"{dataset_name}_validation",
            "experiment_id": args.experiment_id,
            "timestamp": timestamp,
            "debug_mode": args.debug,
            "agent_models": llm_configs,
            "description": "Python DSL workflow",
            "metrics": {
                "score": average_score,
                "metric": "accuracy",
                "success_rate": success_rate,
                "total_problems": total_problems,
                "verified_successful": success_count,
            },
            "output": {
                "trace_file": str(workflow.trace_file),
                "results_dir": str(output_dir)
            }
        }

        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {summary_file}")

        agent_usage = analyze_agent_usage(workflow.trace_file)

        agent_usage_file = output_dir / "agent_usage.json"
        with open(agent_usage_file, 'w', encoding='utf-8') as f:
            json.dump(agent_usage, f, indent=2)

        print("\nAgent Usage Analysis:")
        print(f"  programmer calls: {agent_usage['programmer']['total_calls']}")
        print(f"  refine_solver calls: {agent_usage['refine_solver']['total_calls']}")
        print(f"  detailed_solver calls: {agent_usage['detailed_solver']['total_calls']}")
        print(f"  generate_solver calls: {agent_usage['generate_solver']['total_calls']}")
        print(f"  sc_ensemble calls: {agent_usage['sc_ensemble']['total_calls']}")
        print(f"  Agent usage details saved to: {agent_usage_file}")

        logger.info("Cleaning up resources...")
        if hasattr(workflow, 'programmer') and hasattr(workflow.programmer, 'process_pool'):
            workflow.programmer.process_pool.shutdown(wait=False)
        if hasattr(workflow, 'refine_solver') and hasattr(workflow.refine_solver, 'process_pool'):
            workflow.refine_solver.process_pool.shutdown(wait=False)
        if hasattr(workflow, 'detailed_solver') and hasattr(workflow.detailed_solver, 'process_pool'):
            workflow.detailed_solver.process_pool.shutdown(wait=False)
        if hasattr(workflow, 'generate_solver') and hasattr(workflow.generate_solver, 'process_pool'):
            workflow.generate_solver.process_pool.shutdown(wait=False)

        logger.info("Evaluation complete. Exiting program.")
        print("\nEvaluation complete. Exiting...")

        return average_score

    elif args.task == "hotpotqa":
        # HotpotQA task
        llm_configs = {
            'meta': args.meta_llm or args.llm,
            'answer_generate': args.answer_generate_llm or args.llm,
            'sc_ensemble': args.sc_ensemble_llm or args.llm,
            'format_answer': args.format_answer_llm or args.llm,
        }
        
        logger.info("Starting DSL HotpotQA Workflow evaluation")
        logger.info(f"Raw LLM configurations: {llm_configs}")
        logger.info("Parsed configurations:")
        for agent_name, config in llm_configs.items():
            model, budget = parse_config(config)
            logger.info(f"  {agent_name}: model={model}, budget={budget}")

        file_path = resolve_required_path(args.file_path or "data/hotpotqa_validate.jsonl", label="hotpotqa validate file")
        if args.debug:
            output_dir = profile_root / "debug_dsl_hotpotqa"
            timestamp = "debug"
            logger.info("Debug mode: using DSL output directory")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = profile_root / f"hotpotqa_dsl_agent_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        workflow = DslWorkflowRunner(
            name=f"dsl_hotpotqa{'_debug' if args.debug else ''}",
            llm_configs=llm_configs,
            workflow_type="hotpotqa",
            output_dir=output_dir,
        )

        hotpot_benchmark = HotpotQABenchmark(name="HotpotQA", file_path=file_path, log_path=str(output_dir))
        Path(hotpot_benchmark.log_path).mkdir(parents=True, exist_ok=True)

        logger.info("Running evaluation...")
        average_score, success_count, total_problems, success_rate = await _run_benchmark_with_annotation(
            hotpot_benchmark,
            workflow,
            "HotpotQA",
            max_concurrent_tasks=16 if not args.debug else 1,
        )

        print("\n" + "=" * 80)
        print("DSL HOTPOTQA WORKFLOW EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Experiment ID: {args.experiment_id}")
        print("Dataset: HotpotQA Validation Set")
        print("Workflow: AnswerGenerate x3 -> HotpotQA sc_ensemble -> Format Answer")
        print("\nAgent Configurations:")
        for agent_name, model_config in llm_configs.items():
            model, budget = parse_config(model_config)
            budget_str = f"budget={budget}" if budget is not None else "no budget"
            print(f"  - {agent_name.capitalize()}: {model} ({budget_str})")
        print("\nResults:")
        print(f"  F1: {average_score:.4f} ({average_score*100:.2f}%)")
        print(f"  Success Rate (Verification): {success_rate:.4f} ({success_rate*100:.2f}%)")
        print("\nOutput Files:")
        print(f"  Trace file: {workflow.trace_file}")
        print(f"  Results directory: {output_dir}")
        print("=" * 80)

        summary = {
            "workflow": f"dsl_hotpotqa{'_debug' if args.debug else ''}",
            "workflow_type": "dsl",
            "dataset": "HotpotQA_validation",
            "experiment_id": args.experiment_id,
            "timestamp": timestamp,
            "debug_mode": args.debug,
            "agent_models": llm_configs,
            "description": "Python DSL HotpotQA workflow",
            "metrics": {
                "score": average_score,
                "metric": "f1",
                "success_rate": success_rate,
                "success_count": success_count,
                "total_problems": total_problems,
            },
            "output": {"trace_file": str(workflow.trace_file), "results_dir": str(output_dir)},
        }
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")

        agent_usage = analyze_agent_usage(workflow.trace_file)
        agent_usage_file = output_dir / "agent_usage.json"
        with open(agent_usage_file, "w", encoding="utf-8") as f:
            json.dump(agent_usage, f, indent=2)

        print("\nAgent Usage Analysis:")
        print(
            f"  answer_generate calls: {agent_usage.get('answer_generate', {}).get('total_calls', 0)}"
        )
        print(
            f"  sc_ensemble calls: {agent_usage.get('sc_ensemble', {}).get('total_calls', 0)}"
        )
        print(
            f"  format_answer calls: {agent_usage.get('format_answer', {}).get('total_calls', 0)}"
        )
        print(f"  Agent usage details saved to: {agent_usage_file}")

        logger.info("Evaluation complete. Exiting program.")
        print("\nEvaluation complete. Exiting...")

    elif args.task == "livecodebench":
        # LiveCodeBench task
        llm_configs = {
            'meta': args.meta_llm or args.llm,
            'code_generate': args.code_generate_llm or args.llm,
            'sc_ensemble': args.sc_ensemble_llm or args.llm,
            'test': args.test_llm or args.llm,
            'reflection_test': args.reflection_test_llm or args.llm,
        }
        
        logger.info("Starting DSL LiveCodeBench Workflow evaluation")
        logger.info(f"Raw LLM configurations: {llm_configs}")
        logger.info("Parsed configurations:")
        for agent_name, config in llm_configs.items():
            model, budget = parse_config(config)
            logger.info(f"  {agent_name}: model={model}, budget={budget}")

        file_path = resolve_required_path(
            args.file_path or "data/livecodebench_validate.jsonl",
            label="livecodebench validate file",
        )
        entry_point_file = resolve_required_path(
            getattr(args, "entry_point_file", None) or "data/livecodebench_public_test.jsonl",
            label="livecodebench entry_point_file",
        )
        if args.debug:
            output_dir = profile_root / "debug_dsl_livecodebench"
            timestamp = "debug"
            logger.info("Debug mode: using DSL output directory")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = profile_root / f"livecodebench_dsl_agent_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        workflow = DslWorkflowRunner(
            name=f"dsl_livecodebench{'_debug' if args.debug else ''}",
            llm_configs=llm_configs,
            workflow_type="livecodebench",
            output_dir=output_dir,
        )

        lcb_benchmark = LiveCodeBench(
            name="LiveCodeBench", 
            file_path=file_path, 
            log_path=str(output_dir),
            entry_point_file=entry_point_file
        )
        Path(lcb_benchmark.log_path).mkdir(parents=True, exist_ok=True)

        logger.info("Running evaluation...")
        average_score, success_count, total_problems, success_rate = await _run_benchmark_with_annotation(
            lcb_benchmark,
            workflow,
            "LiveCodeBench",
            max_concurrent_tasks=16 if not args.debug else 1,
        )

        print("\n" + "=" * 80)
        print("DSL LIVECODEBENCH WORKFLOW EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Experiment ID: {args.experiment_id}")
        print("Dataset: LiveCodeBench Validation Set")
        print("Workflow: CodeGenerate x3 -> sc_ensemble -> Test -> Fix (if needed)")
        print("\nAgent Configurations:")
        for agent_name, model_config in llm_configs.items():
            model, budget = parse_config(model_config)
            budget_str = f"budget={budget}" if budget is not None else "no budget"
            print(f"  - {agent_name.capitalize()}: {model} ({budget_str})")
        print("\nResults:")
        print(f"  Pass@1: {average_score:.4f} ({average_score*100:.2f}%)")
        print("\nOutput Files:")
        print(f"  Trace file: {workflow.trace_file}")
        print(f"  Results directory: {output_dir}")
        print("=" * 80)

        summary = {
            "workflow": f"dsl_livecodebench{'_debug' if args.debug else ''}",
            "workflow_type": "dsl",
            "dataset": "LiveCodeBench_validation",
            "experiment_id": args.experiment_id,
            "timestamp": timestamp,
            "debug_mode": args.debug,
            "agent_models": llm_configs,
            "description": "Python DSL LiveCodeBench workflow",
            "metrics": {
                "score": average_score,
                "metric": "pass_at_1",
                "success_rate": success_rate,
                "success_count": success_count,
                "total_problems": total_problems,
            },
            "output": {"trace_file": str(workflow.trace_file), "results_dir": str(output_dir)},
        }
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")

        agent_usage = analyze_agent_usage(workflow.trace_file)
        agent_usage_file = output_dir / "agent_usage.json"
        with open(agent_usage_file, "w", encoding="utf-8") as f:
            json.dump(agent_usage, f, indent=2)

        print("\nAgent Usage Analysis:")
        print(
            f"  code_generate calls: {agent_usage.get('code_generate', {}).get('total_calls', 0)}"
        )
        print(
            f"  sc_ensemble calls: {agent_usage.get('sc_ensemble', {}).get('total_calls', 0)}"
        )
        print(
            f"  test calls: {agent_usage.get('test', {}).get('total_calls', 0)}"
        )
        print(
            f"  reflection_test calls: {agent_usage.get('reflection_test', {}).get('total_calls', 0)}"
        )
        print(f"  Agent usage details saved to: {agent_usage_file}")

        logger.info("Evaluation complete. Exiting program.")
        print("\nEvaluation complete. Exiting...")

        return average_score

def analyze_agent_usage(trace_file: Path) -> dict:
    """Analyze tool usage statistics from trace file (supports both workflow types)"""
    agent_stats = {
        'meta': {'total_calls': 0, 'successful_calls': 0},
        'programmer': {'total_calls': 0, 'successful_calls': 0},
        'refine_solver': {'total_calls': 0, 'successful_calls': 0},
        'detailed_solver': {'total_calls': 0, 'successful_calls': 0},
        'generate_solver': {'total_calls': 0, 'successful_calls': 0},
        'answer_generate': {'total_calls': 0, 'successful_calls': 0},
        'sc_ensemble': {'total_calls': 0, 'successful_calls': 0},
        'format_answer': {'total_calls': 0, 'successful_calls': 0},
        'code_generate': {'total_calls': 0, 'successful_calls': 0},
        'test': {'total_calls': 0, 'successful_calls': 0},
        'reflection_test': {'total_calls': 0, 'successful_calls': 0},
        # BrowseCompPlus agents
        'rewriter': {'total_calls': 0, 'successful_calls': 0},
        'reader_extractor': {'total_calls': 0, 'successful_calls': 0},
        'answer_reviewer': {'total_calls': 0, 'successful_calls': 0},
    }
    
    if not trace_file.exists():
        return agent_stats
    
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            trace = json.loads(line)

            steps = trace.get('steps')
            if isinstance(steps, list):
                for step in steps:
                    agent_name = step.get('agent', '')
                    if agent_name in agent_stats:
                        agent_stats[agent_name]['total_calls'] += 1
                        metadata = step.get('metadata', {})
                        if metadata.get('status') == 'success':
                            agent_stats[agent_name]['successful_calls'] += 1
                continue

            # Handle meta-agent workflow traces
            for step in trace.get('react_steps', []):
                # Count meta-agent decisions (each step is a decision)
                agent_stats['meta']['total_calls'] += 1
                if not step.get('error'):
                    agent_stats['meta']['successful_calls'] += 1

                # Count tool calls (action field contains tool name)
                action = step.get('action', '')
                if action in ['programmer', 'refine_solver', 'detailed_solver', 'generate_solver', 'sc_ensemble']:
                    agent_stats[action]['total_calls'] += 1

                    # Get tool metadata
                    metadata = step.get('tool_metadata', {})
                    if metadata.get('status') == 'success':
                        agent_stats[action]['successful_calls'] += 1
    
    # Calculate success rates
    for agent_name in agent_stats:
        total = agent_stats[agent_name]['total_calls']
        successful = agent_stats[agent_name]['successful_calls']
        agent_stats[agent_name]['success_rate'] = successful / total if total > 0 else 0.0
    
    return agent_stats
