"""
KNN-based Pareto Frontier Router for Workflow Configuration Selection.

This router combines K-Nearest Neighbors with Pareto frontier analysis to:
1. Find similar queries from validation set using embeddings
2. Aggregate sub-agent performance data from neighbors
3. Compute Pareto optimal workflow configurations
4. Return ranked configurations by accuracy threshold

Features:
- Embedding caching for efficiency
- Supports multiple workflow types (math, hotpotqa, livecodebench)
- Pareto frontier computation with configurable accuracy thresholds
- Detailed metadata for debugging and analysis
"""

import numpy as np
import pandas as pd
import torch
import pickle
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from . import Router, RoutingResult, register_router
from workflow_compiler.workflows.dsl_registry import get_workflow_module

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ParetoConfiguration:
    """Represents a Pareto optimal workflow configuration."""
    workflow_id: str
    expected_accuracy: float
    expected_latency: float
    accuracy_threshold: Optional[float] = None
    structure_id: Optional[str] = None
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    subagent_settings: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'expected_accuracy': self.expected_accuracy,
            'expected_latency': self.expected_latency,
            'accuracy_threshold': self.accuracy_threshold,
            'structure_id': self.structure_id,
            'workflow_params': self.workflow_params,
            'subagent_settings': self.subagent_settings,
            'metadata': self.metadata
        }


# ============================================================================
# Embedding and Caching Utilities
# ============================================================================

def text_to_key(text: str) -> str:
    """Generate stable hash key for text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class EmbeddingCache:
    """Manages embedding cache with disk persistence."""
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_file: Path to cache file (pickle)
        """
        self.cache_file = cache_file
        self.cache: Dict[str, np.ndarray] = {}
        
        if cache_file and Path(cache_file).exists():
            self.load()
    
    def load(self):
        """Load cache from disk."""
        if not self.cache_file:
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            self.cache = {k: np.asarray(v) for k, v in self.cache.items()}
            logger.info(f"Loaded {len(self.cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def save(self):
        """Save cache to disk."""
        if not self.cache_file:
            return
        
        try:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = text_to_key(text)
        return self.cache.get(key)
    
    def put(self, text: str, embedding: np.ndarray):
        """Add embedding to cache."""
        key = text_to_key(text)
        self.cache[key] = embedding
    
    def has(self, text: str) -> bool:
        """Check if text is in cache."""
        key = text_to_key(text)
        return key in self.cache


class QueryEmbedder:
    """Embed queries using transformer models."""
    
    def __init__(
        self,
        model_name: str = 'allenai/longformer-base-4096',
        max_length: int = 4096,
        device: Optional[str] = None
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            device: Device ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.tokenizer = None
        self.model = None
        
        logger.info(f"QueryEmbedder: {model_name} on {self.device}")
    
    def _init_model(self):
        """Lazy initialization of model."""
        if self.model is not None:
            return
        
        from transformers import AutoTokenizer, AutoModel
        
        logger.info(f"Loading embedder: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed single text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector (numpy array)
        """
        self._init_model()
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Embed batch of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size
        
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        self._init_model()
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


# ============================================================================
# Pareto Frontier Utilities
# ============================================================================

def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find Pareto efficient points.
    
    Args:
        costs: Array of shape (n_points, n_objectives) to minimize
    
    Returns:
        Boolean mask of Pareto efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep points that are not dominated
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )
            is_efficient[i] = True
    return is_efficient


def filter_pareto_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to Pareto optimal rows.
    
    Args:
        df: DataFrame with 'accuracy' and 'latency' columns
    
    Returns:
        Filtered DataFrame with only Pareto optimal rows
    """
    if len(df) == 0:
        return df
    
    costs = np.column_stack([
        df['latency'].values,
        -df['accuracy'].values
    ])
    
    pareto_mask = is_pareto_efficient(costs)
    return df[pareto_mask].reset_index(drop=True)


# ============================================================================
# KNN Pareto Router
# ============================================================================

@register_router("knn")
class KNNRouter(Router):
    """
    KNN-based Pareto Frontier Router.
    
    Routes queries to Pareto optimal workflow configurations based on:
    1. Finding K nearest neighbors from validation set
    2. Aggregating sub-agent performance from neighbors
    3. Computing Pareto frontier of workflow configurations
    4. Selecting configurations by accuracy threshold
    """
    
    def __init__(
        self,
        name: str = "knn",
        k: int = 10,
        embedding_model: str = 'allenai/longformer-base-4096',
        max_length: int = 4096,
        embedding_cache_file: Optional[str] = None,
        embedding_batch_size: int = 8,
        accuracy_thresholds: Optional[List[float]] = None,
        search_space: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize KNN router.
        
        Args:
            name: Router name
            k: Number of nearest neighbors
            embedding_model: Model for query embeddings
            max_length: Max sequence length for embeddings
            embedding_cache_file: Path to embedding cache file
            embedding_batch_size: Batch size for embedding validation queries
            accuracy_thresholds: List of accuracy thresholds (default: [0.8, 0.85, 0.9, 0.95, 0.99])
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
        self.k = k
        self.embedding_model = embedding_model
        self.max_length = max_length
        self.embedding_batch_size = embedding_batch_size
        
        if accuracy_thresholds is None:
            accuracy_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
        self.accuracy_thresholds = accuracy_thresholds
        self.search_space = search_space
        
        # Initialize components
        self.embedder = QueryEmbedder(
            model_name=embedding_model,
            max_length=max_length
        )
        self.embedding_cache = EmbeddingCache(embedding_cache_file)
        
        # Training data
        self.query_data_table: Optional[Dict[str, Any]] = None
        self.knn_index = None
        self.query_id_list: Optional[List[str]] = None
        self.validation_embeddings: Optional[np.ndarray] = None
        
        logger.info(f"KNNRouter initialized (k={k}, thresholds={accuracy_thresholds})")
    
    def fit(self, training_data: List[Dict[str, Any]], **kwargs):
        """
        Fit router on validation data.
        
        Args:
            training_data: List of validation query data with format:
                [
                    {
                        'query_id': str,
                        'query_text': str,
                        'agents': {
                            'agent_name': {
                                'setting_name': {
                                    'accuracy': float,
                                    'latency': float
                                }
                            }
                        }
                    },
                    ...
                ]
            **kwargs: Additional parameters
        """
        logger.info(f"Fitting KNN router on {len(training_data)} validation queries")
        
        # Convert list to query_data_table format
        self.query_data_table = {
            item['query_id']: item
            for item in training_data
        }
        
        # Build KNN index
        self._build_knn_index()
        
        logger.info(f"Router fitted with {len(self.query_data_table)} queries")
    
    def fit_from_query_table(self, query_data_table: Dict[str, Any]):
        """
        Fit router from consolidated query data table.
        
        Args:
            query_data_table: Dict mapping query_id -> query data
        """
        logger.info(f"Fitting from query table with {len(query_data_table)} queries")
        self.query_data_table = query_data_table
        self._build_knn_index()
    
    def _build_knn_index(self):
        """Build KNN index from validation queries."""
        from sklearn.neighbors import NearestNeighbors
        
        if not self.query_data_table:
            raise ValueError("No query data table available")
        
        logger.info("Building KNN index...")
        
        # Extract query texts and IDs
        self.query_id_list = list(self.query_data_table.keys())
        query_texts = [
            self.query_data_table[qid]['query_text']
            for qid in self.query_id_list
        ]
        
        # Embed queries (use cache when available)
        embeddings = []
        missing_texts = []
        missing_indices = []
        
        for i, text in enumerate(query_texts):
            cached_emb = self.embedding_cache.get(text)
            if cached_emb is not None:
                embeddings.append(cached_emb)
            else:
                embeddings.append(None)
                missing_texts.append(text)
                missing_indices.append(i)
        
        # Embed missing queries
        if missing_texts:
            logger.info(f"Embedding {len(missing_texts)} missing queries...")
            new_embeddings = self.embedder.embed_batch(missing_texts, batch_size=self.embedding_batch_size)
            
            for idx, emb in zip(missing_indices, new_embeddings):
                embeddings[idx] = emb
                self.embedding_cache.put(query_texts[idx], emb)
            
            # Save cache
            self.embedding_cache.save()
        else:
            logger.info("All validation embeddings found in cache")
        
        self.validation_embeddings = np.vstack(embeddings)
        
        # Build KNN index
        self.knn_index = NearestNeighbors(
            n_neighbors=min(self.k, len(self.query_id_list)),
            metric='cosine'
        )
        self.knn_index.fit(self.validation_embeddings)
        
        logger.info(f"KNN index built with {len(self.query_id_list)} queries")
    
    def route(
        self,
        query: Dict[str, Any],
        candidate_workflows: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        workflow_type: Optional[str] = None,
        **kwargs
    ) -> RoutingResult:
        """
        Route query to Pareto optimal workflow configurations.
        
        Args:
            query: Query dict with 'query_text' or 'problem'
            candidate_workflows: Not used (computes all possible workflows)
            top_k: Number of configurations to return
            workflow_type: Workflow type ('math', 'hotpotqa', 'livecodebench')
            **kwargs: Additional parameters
        
        Returns:
            RoutingResult with Pareto optimal configurations
        """
        if self.knn_index is None:
            raise RuntimeError("Router not fitted. Call fit() first.")
        
        if workflow_type is None:
            workflow_type = kwargs.get('workflow_type', 'math')
        
        # Extract query text
        query_text = query.get('query_text') or query.get('problem', '')
        query_id = query.get('query_id') or query.get('unique_id') or query.get('_id')
        
        # Embed query (use cache)
        cached_emb = self.embedding_cache.get(query_text)
        if cached_emb is not None:
            query_embedding = cached_emb
        else:
            query_embedding = self.embedder.embed(query_text)
            self.embedding_cache.put(query_text, query_embedding)
            self.embedding_cache.save()
        
        # Find K nearest neighbors
        distances, indices = self.knn_index.kneighbors(
            query_embedding.reshape(1, -1),
            n_neighbors=min(self.k, len(self.query_id_list))
        )
        
        neighbor_query_ids = [self.query_id_list[idx] for idx in indices[0]]
        
        # Compute Pareto frontier from neighbors
        pareto_configs = self._compute_pareto_frontier(
            neighbor_query_ids,
            workflow_type,
            query_id
        )
        
        # Build ranking
        ranking = [
            (config.workflow_id, config.expected_accuracy)
            for config in pareto_configs[:top_k]
        ]
        
        metadata = {
            'method': 'knn',
            'k': self.k,
            'workflow_type': workflow_type,
            'neighbor_ids': neighbor_query_ids,
            'neighbor_distances': distances[0].tolist(),
            'pareto_configs': [config.to_dict() for config in pareto_configs],
            'num_pareto_configs': len(pareto_configs)
        }
        
        return RoutingResult(ranking=ranking, metadata=metadata)
    
    def _compute_pareto_frontier(
        self,
        neighbor_query_ids: List[str],
        workflow_type: str,
        query_id: Optional[str] = None
    ) -> List[ParetoConfiguration]:
        """
        Compute Pareto frontier from neighbors.
        
        Args:
            neighbor_query_ids: List of neighbor query IDs
            workflow_type: Workflow type
            query_id: Current query ID (for metadata)
        
        Returns:
            List of ParetoConfiguration objects
        """
        workflow_module = get_workflow_module(workflow_type)

        # Aggregate neighbor data
        agent_dfs = self._aggregate_neighbor_data(neighbor_query_ids, workflow_module)
        
        # Pre-filter each agent to Pareto optimal configs
        for df_name in agent_dfs:
            agent_dfs[df_name] = filter_pareto_optimal(agent_dfs[df_name])
        
        metadata = {
            "search_space": self.search_space,
        }
        workflow_df = workflow_module.compute_configs(agent_dfs, metadata)

        if workflow_df is None or workflow_df.empty:
            return []
        
        # Find Pareto frontier
        costs = np.column_stack([
            workflow_df['workflow_latency'].values,
            -workflow_df['workflow_accuracy'].values
        ])
        
        pareto_mask = is_pareto_efficient(costs)
        pareto_df = workflow_df[pareto_mask].sort_values('workflow_latency').reset_index(drop=True)
        
        # Select configurations by accuracy threshold
        pareto_configs = []
        selected_indices = set()
        
        for threshold in sorted(self.accuracy_thresholds, reverse=True):
            eligible = pareto_df[pareto_df['workflow_accuracy'] >= threshold]
            
            if len(eligible) > 0:
                min_latency_idx = eligible['workflow_latency'].idxmin()
                
                if min_latency_idx not in selected_indices:
                    selected_indices.add(min_latency_idx)
                    row = pareto_df.loc[min_latency_idx]
                    
                    config = self._build_pareto_config(
                        row, workflow_type, query_id, threshold
                    )
                    pareto_configs.append(config)
        
        # Add highest accuracy config if not already included
        if len(pareto_df) > 0 and len(pareto_configs) == 0:
            row = pareto_df.loc[pareto_df['workflow_accuracy'].idxmax()]
            config = self._build_pareto_config(row, workflow_type, query_id, None)
            pareto_configs.append(config)
        
        return pareto_configs
    
    def _aggregate_neighbor_data(
        self,
        neighbor_query_ids: List[str],
        workflow_module: Any,
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate sub-agent performance from neighbors.
        
        Args:
            neighbor_query_ids: List of neighbor query IDs
            workflow_module: Workflow module used for sub-agent alias normalization
        
        Returns:
            Dict of DataFrames for each sub-agent
        """
        # Collect data from all neighbors
        all_records = []
        
        for query_id in neighbor_query_ids:
            query_data = self.query_data_table[query_id]
            
            for agent_name, settings_data in query_data['agents'].items():
                for setting_name, metrics in settings_data.items():
                    all_records.append({
                        'query_id': query_id,
                        'subagent': agent_name,
                        'setting': setting_name,
                        'accuracy': metrics['accuracy'],
                        'latency': metrics['latency']
                    })
        
        df = pd.DataFrame(all_records)
        
        # Aggregate by subagent and setting
        df_agg = df.groupby(['subagent', 'setting']).agg({
            'accuracy': 'mean',
            'latency': 'mean'
        }).reset_index()
        
        # Split by sub-agent and normalize names through workflow aliases.
        agent_dfs = {
            subagent: df_agg[df_agg['subagent'] == subagent][['setting', 'accuracy', 'latency']].reset_index(drop=True)
            for subagent in df_agg['subagent'].unique()
        }
        return workflow_module.normalize_subagent_stats(agent_dfs)
    
    def _build_pareto_config(
        self,
        row: pd.Series,
        workflow_type: str,
        query_id: Optional[str],
        accuracy_threshold: Optional[float]
    ) -> ParetoConfiguration:
        """Build ParetoConfiguration from DataFrame row."""

        workflow_params: Dict[str, Any] = {}
        subagent_settings = {}

        if 'total_branches' in row and pd.notna(row.get('total_branches')):
            workflow_params['total_branches'] = int(row['total_branches'])
        if 'is_full' in row and pd.notna(row.get('is_full')):
            workflow_params['is_full'] = bool(row['is_full'])
        for col in row.index:
            if col.endswith('_count') and pd.notna(row[col]):
                workflow_params[col] = int(row[col])

        for col in row.index:
            if col.endswith('_setting') and pd.notna(row[col]):
                agent_name = col.replace('_setting', '')
                subagent_settings[agent_name] = str(row[col])
        
        # Generate workflow ID
        workflow_id = self._format_workflow_id(row, workflow_type)
        
        return ParetoConfiguration(
            workflow_id=workflow_id,
            expected_accuracy=float(row['workflow_accuracy']),
            expected_latency=float(row['workflow_latency']),
            accuracy_threshold=accuracy_threshold,
            structure_id=str(row.get('structure_id', '')),
            workflow_params=workflow_params,
            subagent_settings=subagent_settings,
            metadata={
                'query_id': query_id,
                'workflow_type': workflow_type
            }
        )
    
    def _format_workflow_id(self, row: pd.Series, workflow_type: str) -> str:
        """Format workflow configuration into readable ID string."""
        del workflow_type
        agents = []

        for col in sorted(row.index):
            if not col.endswith('_setting'):
                continue
            if pd.isna(row.get(col)):
                continue
            agent = col.replace('_setting', '')
            agents.append(f"{agent}={row[col]}")

        return "|".join(agents) if agents else "default"
    
    def save(self, path: str):
        """Save router state to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'k': self.k,
            'embedding_model': self.embedding_model,
            'max_length': self.max_length,
            'accuracy_thresholds': self.accuracy_thresholds,
            'query_data_table': self.query_data_table,
            'validation_embeddings': self.validation_embeddings,
            'query_id_list': self.query_id_list,
            'config': self.config
        }
        
        with open(save_dir / 'knn_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        # Save embedding cache
        self.embedding_cache.save()
        
        logger.info(f"KNN router saved to {path}")
    
    def load(self, path: str):
        """Load router state from disk."""
        load_path = Path(path) / 'knn_state.pkl'
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.k = state['k']
        self.embedding_model = state['embedding_model']
        self.max_length = state['max_length']
        self.accuracy_thresholds = state['accuracy_thresholds']
        self.query_data_table = state['query_data_table']
        self.validation_embeddings = state['validation_embeddings']
        self.query_id_list = state['query_id_list']
        self.config = state.get('config', {})
        
        # Rebuild KNN index
        from sklearn.neighbors import NearestNeighbors
        
        self.knn_index = NearestNeighbors(
            n_neighbors=min(self.k, len(self.query_id_list)),
            metric='cosine'
        )
        self.knn_index.fit(self.validation_embeddings)
        
        logger.info(f"KNN router loaded from {path}")
