"""
Unit tests for workflow_compiler.core.analysis module.
"""

import pytest
import numpy as np
from workflow_compiler.core.analysis import (
    MODEL_TO_HF_NAME,
    get_hf_model_name,
    extract_model_name,
    calculate_latency,
    load_latency_data,
    get_default_latency_data,
    compute_pareto_frontier
)


class TestModelNameMapping:
    """Tests for model name mapping functions."""
    
    def test_get_hf_model_name(self):
        """Test HF name lookup."""
        assert get_hf_model_name('qwen3-4b') == 'Qwen/Qwen3-4B'
        assert get_hf_model_name('qwen3-32b') == 'Qwen/Qwen3-32B'
        assert get_hf_model_name('unknown') == 'unknown'
    
    def test_extract_model_name_basic(self):
        """Test basic model name extraction."""
        assert extract_model_name('qwen3-4b') == 'qwen3-4b'
        assert extract_model_name('qwen3-4b_budget_1000') == 'qwen3-4b'
        assert extract_model_name('qwen3-8b_budget_unlimited') == 'qwen3-8b'
    
    def test_extract_model_name_with_hf(self):
        """Test HF name extraction."""
        assert extract_model_name('qwen3-4b', return_hf_name=True) == 'Qwen/Qwen3-4B'
        assert extract_model_name('qwen3-4b_budget_1000', return_hf_name=True) == 'Qwen/Qwen3-4B'
    
    def test_extract_model_name_with_budget(self):
        """Test budget extraction."""
        model, budget = extract_model_name('qwen3-4b_budget_1000', return_budget=True)
        assert model == 'qwen3-4b'
        assert budget == 1000
        
        model, budget = extract_model_name('qwen3-8b_budget_unlimited', return_budget=True)
        assert model == 'qwen3-8b'
        assert budget == -1
        
        model, budget = extract_model_name('qwen3-4b', return_budget=True)
        assert model == 'qwen3-4b'
        assert budget == 0
    
    def test_extract_model_name_non_string(self):
        """Test handling of non-string inputs."""
        assert extract_model_name(True) == 'True'
        assert extract_model_name(123) == '123'
        
        model, budget = extract_model_name(True, return_budget=True)
        assert model == 'True'
        assert budget == 0


class TestLatencyCalculation:
    """Tests for latency calculation functions."""
    
    def test_get_default_latency_data(self):
        """Test default latency data generation."""
        data = get_default_latency_data()
        assert 'Qwen/Qwen3-4B' in data
        assert 'prefill_latency_per_token' in data['Qwen/Qwen3-4B']
        assert 'decode_latency_per_token' in data['Qwen/Qwen3-4B']
    
    def test_calculate_latency(self):
        """Test latency calculation."""
        latency_data = {
            'Qwen/Qwen3-4B': {
                'prefill_latency_per_token': 0.0002,
                'decode_latency_per_token': 0.002
            }
        }
        
        # Test with base model name
        latency = calculate_latency(100, 50, 'qwen3-4b', latency_data)
        expected = 100 * 0.0002 + 50 * 0.002  # 0.02 + 0.1 = 0.12
        assert abs(latency - expected) < 1e-6
        
        # Test with budget string
        latency = calculate_latency(100, 50, 'qwen3-4b_budget_1000', latency_data)
        assert abs(latency - expected) < 1e-6
    
    def test_calculate_latency_unknown_model(self):
        """Test latency calculation with unknown model."""
        latency_data = {}
        latency = calculate_latency(100, 50, 'unknown_model', latency_data)
        assert latency == 0.0


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""
    
    def test_pareto_basic(self):
        """Test basic Pareto frontier."""
        points = [
            (1.0, 5.0),
            (2.0, 4.0),
            (3.0, 3.0),
            (4.0, 2.0),
            (2.5, 4.5)  # Dominated by (2.0, 4.0) when minimizing x, maximizing y
        ]
        
        # Minimize x, maximize y
        frontier = compute_pareto_frontier(points, maximize_x=False, maximize_y=True)
        
        # Should include (1.0, 5.0), (2.0, 4.0), (2.5, 4.5) actually not dominated
        # Let's check: (1.0, 5.0) dominates (2.0, 4.0) in terms of (min x, max y)
        # Actually (1.0, 5.0) has lower x and higher y, so dominates all
        assert len(frontier) >= 1
        assert (1.0, 5.0) in frontier
    
    def test_pareto_empty(self):
        """Test with empty points."""
        frontier = compute_pareto_frontier([])
        assert frontier == []
    
    def test_pareto_single(self):
        """Test with single point."""
        points = [(1.0, 2.0)]
        frontier = compute_pareto_frontier(points)
        assert frontier == points


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
