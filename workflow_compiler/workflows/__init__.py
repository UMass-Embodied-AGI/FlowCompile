"""
Workflow subsystem for FlowCompile.

Provides:
- Workflow base class
- Workflow registry for discovery
- Fixed workflow implementations (math, hotpotqa, code)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Base Class
# ============================================================================

class Workflow(ABC):
    """
    Abstract base class for workflows.
    
    A workflow orchestrates multiple agents/steps to solve a problem.
    """
    
    def __init__(
        self,
        name: str,
        llm_config: Optional[Dict] = None,
        dataset: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize workflow.
        
        Args:
            name: Workflow name
            llm_config: LLM configuration
            dataset: Dataset name
            **kwargs: Additional workflow-specific parameters
        """
        self.name = name
        self.llm_config = llm_config
        self.dataset = dataset
        self.config = kwargs
    
    @abstractmethod
    async def __call__(self, problem: Any) -> Any:
        """
        Execute workflow on a problem.
        
        Args:
            problem: Problem input (dict, string, etc.)
        
        Returns:
            Workflow output
        """
        pass
    
    def get_structure_description(self) -> Dict[str, Any]:
        """
        Get description of workflow structure.
        
        Returns:
            Dictionary describing workflow structure
        """
        return {
            'name': self.name,
            'dataset': self.dataset
        }


# ============================================================================
# Workflow Registry
# ============================================================================

_WORKFLOW_REGISTRY: Dict[str, Callable[..., Workflow]] = {}


def register_workflow(name: str):
    """
    Decorator to register a workflow class.
    
    Usage:
        @register_workflow("math_solver")
        class MathSolverWorkflow(Workflow):
            ...
    
    Args:
        name: Workflow name for lookup
    """
    def decorator(workflow_class: type) -> type:
        if not issubclass(workflow_class, Workflow):
            raise TypeError(f"Workflow class must inherit from Workflow: {workflow_class}")
        
        _WORKFLOW_REGISTRY[name] = workflow_class
        logger.debug(f"Registered workflow: {name} -> {workflow_class.__name__}")
        return workflow_class
    
    return decorator


def get_workflow(name: str, **kwargs) -> Workflow:
    """
    Get a workflow instance by name.
    
    Args:
        name: Workflow name
        **kwargs: Parameters to pass to workflow constructor
    
    Returns:
        Workflow instance
    
    Raises:
        ValueError: If workflow not found
    
    Example:
        >>> workflow = get_workflow('math_solver', llm_config={'model': 'gpt-4'})
        >>> result = await workflow(problem)
    """
    if name not in _WORKFLOW_REGISTRY:
        available = ', '.join(_WORKFLOW_REGISTRY.keys())
        raise ValueError(
            f"Workflow '{name}' not found. Available workflows: {available}"
        )
    
    workflow_class = _WORKFLOW_REGISTRY[name]
    return workflow_class(name=name, **kwargs)


def list_workflows() -> List[str]:
    """Get list of registered workflow names."""
    return list(_WORKFLOW_REGISTRY.keys())


# ============================================================================
# Factory Function
# ============================================================================

def create_workflow(
    workflow_type: str,
    structure_id: Optional[str] = None,
    llm_configs: Optional[Dict] = None,
    **kwargs
) -> Workflow:
    """
    Factory function to create workflow instances.
    
    Args:
        workflow_type: Type of workflow ('math', 'hotpotqa', 'code', etc.)
        structure_id: Optional structure identifier
        llm_configs: Dictionary of agent -> LLM config mappings
        **kwargs: Additional parameters
    
    Returns:
        Workflow instance
    """
    raise ValueError(
        f"Workflow '{workflow_type}' is now defined via Python DSL; "
        "use the DSL runtime path instead of create_workflow."
    )


# ============================================================================
# Workflow Re-exports
# ============================================================================

__all__ = [
    'Workflow',
    'register_workflow',
    'get_workflow',
    'list_workflows',
    'create_workflow'
]
