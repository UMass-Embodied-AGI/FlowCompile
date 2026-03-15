"""
Unit tests for workflow_compiler.core.llm.config module.
"""

import pytest
import tempfile
from pathlib import Path
from workflow_compiler.core.llm.config import (
    load_config,
    load_model_alias_to_hf_name_map,
    parse_llm_config,
    create_experiment_config,
    ExperimentConfig,
    merge_configs
)


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_model_alias_to_hf_name_map(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "models:",
                    "  qwen3-4b:",
                    "    hf_model_name: Qwen/Qwen3-4B",
                    "  worker-route:",
                    "    model: qwen3-8b",
                    "    hf_model_name: Qwen/Qwen3-8B",
                    "  no-hf:",
                    "    model: missing-hf",
                ]
            ),
            encoding="utf-8",
        )

        mapping = load_model_alias_to_hf_name_map(str(config_path))

        assert mapping["qwen3-4b"] == "Qwen/Qwen3-4B"
        assert mapping["worker-route"] == "Qwen/Qwen3-8B"
        assert mapping["qwen3-8b"] == "Qwen/Qwen3-8B"
        assert "missing-hf" not in mapping
    
    def test_parse_llm_config(self):
        """Test LLM config string parsing."""
        config = parse_llm_config('qwen3-4b_budget_1000')
        assert config['model'] == 'qwen3-4b'
        assert config['budget'] == 1000
        
        config = parse_llm_config('qwen3-8b_budget_unlimited')
        assert config['model'] == 'qwen3-8b'
        assert config['budget'] == -1
    
    def test_create_experiment_config(self):
        """Test experiment config creation."""
        config = create_experiment_config(
            name='test_exp',
            benchmark='math',
            workflow_type='fixed',
            output_dir='results/test'
        )
        
        assert isinstance(config, ExperimentConfig)
        assert config.name == 'test_exp'
        assert config.benchmark == 'math'
        assert config.workflow_type == 'fixed'
        assert config.output_dir == 'results/test'


class TestConfigMerging:
    """Tests for config merging."""
    
    def test_merge_configs_basic(self):
        """Test basic config merging."""
        base = {
            'model': 'qwen3-4b',
            'temperature': 0.0
        }
        override = {
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        merged = merge_configs(base, override)
        
        assert merged['model'] == 'qwen3-4b'
        assert merged['temperature'] == 0.7
        assert merged['max_tokens'] == 1000
    
    def test_merge_configs_nested(self):
        """Test nested config merging."""
        base = {
            'llm': {
                'model': 'qwen3-4b',
                'temperature': 0.0
            }
        }
        override = {
            'llm': {
                'temperature': 0.7
            }
        }
        
        merged = merge_configs(base, override)
        
        assert merged['llm']['model'] == 'qwen3-4b'
        assert merged['llm']['temperature'] == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
