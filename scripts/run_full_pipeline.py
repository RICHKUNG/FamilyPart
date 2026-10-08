#!/usr/bin/env python3
"""
Full pipeline runner for My3DIS experiments (Aggregation + Inference).

This script orchestrates the complete workflow from My3DIS tracking outputs
to open-vocabulary instance segmentation predictions:

1. Aggregation: Load multi-level tracking outputs and aggregate to 3D proposals
2. Feature Extraction: Extract SigLIP features for each proposal
3. Inference: Run OV-3DIS inference with multi-instance handling
4. (Optional) Evaluation: Compute mAP using Search3D protocol

Usage:
    python scripts/run_full_pipeline.py --config configs/inference/experiment_pipeline.yaml

    # Override specific parameters
    python scripts/run_full_pipeline.py \
        --config configs/inference/experiment_pipeline.yaml \
        --exp-dir /path/to/experiment \
        --skip-aggregation  # Skip if aggregation already done

Features:
- Unified YAML configuration for all pipeline stages
- Automatic path management (output dirs default to exp_dir subdirectories)
- Stage toggles (enable/disable aggregation, inference, evaluation)
- Detailed logging and progress tracking
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3DprojToSiglip"))

import argparse
import json
import logging
import time
import yaml
from typing import Dict, Optional, List, Tuple
from collections import Counter

import torch

# My3DIS imports
from my3dis.aggregation import AggregationPipeline
from my3dis.inference import (
    InferencePipeline,
    SigLIPFeatureExtractor,
    OptimizedSigLIPFeatureExtractor,  # 新增：优化版本
)
from my3dis.inference.config import InferenceConfig
from my3dis.utils import load_scene_pointcloud, save_mask_as_ply

# 全局开关：启用优化版本（设为 False 可回退到原始版本）
USE_OPTIMIZED_EXTRACTION = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_model_preset(config: Dict, model_type: Optional[str] = None) -> Dict:
    """
    Apply model preset configuration based on model type.

    This function:
    1. Selects the model preset (siglip, laion_clip, etc.)
    2. Populates siglip_model, scale_semantic_score, apply_softmax across all sections
    3. Generates output directory names with model_id and pooling

    Args:
        config: Full configuration dictionary
        model_type: Override model type (from CLI --model argument)

    Returns:
        Updated configuration dictionary
    """
    # Check if model config exists
    model_config = config.get('model', {})
    if not model_config:
        logger.info("No model config section found, skipping preset application")
        return config

    # Determine which model type to use
    if model_type:
        selected_type = model_type
    else:
        selected_type = model_config.get('type', 'siglip')

    # Get presets
    presets = model_config.get('presets', {})
    if selected_type not in presets:
        logger.warning(f"Model type '{selected_type}' not found in presets, using siglip")
        selected_type = 'siglip'

    preset = presets[selected_type]
    model_name = preset['model_name']
    model_id = preset['model_id']
    scale_score = preset['scale_semantic_score']
    apply_softmax = preset['apply_softmax']
    text_template = preset.get('text_template', 'in_room')

    logger.info(f"Applying model preset: {selected_type}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Model ID: {model_id}")
    logger.info(f"  Scale: {scale_score}, Softmax: {apply_softmax}")

    # Get pooling mode
    pooling_mode = config.get('aggregation', {}).get('pooling_mode', 'average')

    # Generate output directory suffix
    output_suffix = f"{model_id}_{pooling_mode}"

    # Store model info for reference
    config['_model_info'] = {
        'type': selected_type,
        'model_name': model_name,
        'model_id': model_id,
        'output_suffix': output_suffix,
    }

    # Apply to aggregation
    if 'aggregation' in config:
        config['aggregation']['siglip_model'] = model_name

    # Apply to retrieval
    if 'inference' in config and 'retrieval' in config['inference']:
        config['inference']['retrieval']['scale_semantic_score'] = scale_score
        config['inference']['retrieval']['apply_softmax'] = apply_softmax
        config['inference']['retrieval']['text_template'] = text_template

    # Apply to all strategies
    for strategy in ['independent', 'hierarchical', 'exhaustive_pairing']:
        if 'inference' in config and strategy in config['inference']:
            config['inference'][strategy]['siglip_model'] = model_name
            config['inference'][strategy]['scale_semantic_score'] = scale_score
            config['inference'][strategy]['apply_softmax'] = apply_softmax
            if 'text_template' not in config['inference'][strategy]:
                config['inference'][strategy]['text_template'] = text_template

    return config


def setup_output_directories(config: Dict, exp_dir: Path) -> Dict:
    """
    Setup output directories with model-based naming.

    Directory format: {exp_dir}/aggregation_output_{model_id}_{pooling}/

    Args:
        config: Configuration dictionary (must have _model_info applied)
        exp_dir: Experiment directory path

    Returns:
        Updated configuration with resolved output paths
    """
    # Get model info
    model_info = config.get('_model_info', {})
    output_suffix = model_info.get('output_suffix', 'default')

    # Resolve aggregation output directory
    if config['experiment'].get('aggregation_output_dir') is None:
        config['experiment']['aggregation_output_dir'] = str(
            exp_dir / f'aggregation_output_{output_suffix}'
        )

    # Resolve inference output directory
    if config['experiment'].get('inference_output_dir') is None:
        config['experiment']['inference_output_dir'] = str(
            exp_dir / f'inference_output_{output_suffix}'
        )

    return config


def setup_paths(config: Dict) -> Dict:
    """
    Setup paths for aggregation and inference outputs.

    Returns updated config with resolved paths.
    """
    exp_dir = Path(config['experiment']['exp_dir'])

    # Auto-detect levels if not specified
    if config['experiment'].get('levels') is None:
        detected_levels = []
        for level_dir in sorted(exp_dir.glob('level_*')):
            if level_dir.is_dir():
                level_num = int(level_dir.name.split('_')[1])
                detected_levels.append(level_num)

        if detected_levels:
            config['experiment']['levels'] = detected_levels
            logger.info(f"Auto-detected levels from exp_dir: {detected_levels}")
        else:
            # Fallback to common default
            config['experiment']['levels'] = [2, 4, 6]
            logger.warning(f"Could not auto-detect levels, using default: [2, 4, 6]")

    # Use model-based directory naming if model info is available
    if '_model_info' in config:
        config = setup_output_directories(config, exp_dir)
    else:
        # Fallback to legacy naming
        if config['experiment'].get('aggregation_output_dir') is None:
            config['experiment']['aggregation_output_dir'] = str(exp_dir / 'aggregation_output')
        if config['experiment'].get('inference_output_dir') is None:
            config['experiment']['inference_output_dir'] = str(exp_dir / 'inference_output')

    return config


def _dict_to_inference_config(inf_dict: Dict) -> InferenceConfig:
    """
    Convert inference config dictionary to InferenceConfig object.

    This is a workaround since InferenceConfig.from_yaml expects a file path,
    but we're working with an already-loaded dict.
    """
    from my3dis.inference.config import (
        RetrievalConfig,
        PairingConfig,
        NMSConfig,
        ScoringConfig,
        FormatterConfig,
        IndependentStrategyConfig,
        HierarchicalStrategyConfig,
        ExhaustivePairingConfig,
    )

    config = InferenceConfig()

    if 'strategy' in inf_dict:
        config.strategy = inf_dict['strategy']

    if 'retrieval' in inf_dict:
        config.retrieval = RetrievalConfig(**inf_dict['retrieval'])

    if 'pairing' in inf_dict:
        config.pairing = PairingConfig(**inf_dict['pairing'])

    if 'nms' in inf_dict:
        config.nms = NMSConfig(**inf_dict['nms'])

    if 'scoring' in inf_dict:
        config.scoring = ScoringConfig(**inf_dict['scoring'])

    if 'formatter' in inf_dict:
        config.formatter = FormatterConfig(**inf_dict['formatter'])

    if 'independent' in inf_dict:
        config.independent = IndependentStrategyConfig(**inf_dict['independent'])

    if 'hierarchical' in inf_dict:
        config.hierarchical = HierarchicalStrategyConfig(**inf_dict['hierarchical'])

    if 'exhaustive_pairing' in inf_dict:
        config.exhaustive_pairing = ExhaustivePairingConfig(**inf_dict['exhaustive_pairing'])

    return config


def load_scene_label_distribution(gt_root: Optional[str], scene_name: str) -> Optional[Counter]:
    """
    Load per-scene label distribution from Search3D GT annotations.

    The GT files store per-point joint semantic ids encoded as
    label_id * 1000 + instance_id. We only care about the label ids.
    """
    if not gt_root:
        return None

    gt_file = Path(gt_root) / f"{scene_name}_obj_part_inst.txt"
    if not gt_file.exists():
        logger.warning(f"GT file not found for scene {scene_name}: {gt_file}")
        return None

    label_counts: Counter = Counter()
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = int(line)
            except ValueError:
                continue
            if value <= 0:
                continue
            label_id = value // 1000
            label_counts[label_id] += 1

    if not label_counts:
        logger.warning(f"No positive labels found in GT for scene {scene_name} ({gt_file})")
        return None

    return label_counts

def run_aggregation_stage(config: Dict) -> Dict[str, str]:
    """
    Run aggregation stage: 2D masks → 3D proposals + features.

    Returns:
        Dictionary mapping level -> NPZ file path
    """
    logger.info("="*80)
    logger.info("STAGE 1: AGGREGATION (2D → 3D)")
    logger.info("="*80)

    exp_config = config['experiment']
    agg_config = config['aggregation']

    # Parse per-level area fractions if provided
    area_fraction_per_level = agg_config.get('area_fraction_per_level', None)
    if area_fraction_per_level:
        # Convert string keys to int keys
        area_fraction_per_level = {int(k): v for k, v in area_fraction_per_level.items()}

    # Initialize aggregation pipeline
    pipeline = AggregationPipeline(
        exp_dir=exp_config['exp_dir'],
        scene_path=exp_config['scene_path'],
        iou_threshold=agg_config['iou_threshold'],
        area_fraction=agg_config['area_fraction'],
        device=exp_config['device'],
        config={
            "uni3dis": {
                "vis_depth_threshold": agg_config['projection']['vis_depth_threshold'],
                "frequency": agg_config['projection']['frequency']
            },
            "network2d": {
                "masklet": {
                    "resize_ratio": agg_config['projection']['resize_ratio']
                }
            }
        },
        pooling_mode=agg_config['pooling_mode'],
        area_fraction_per_level=area_fraction_per_level,
    )

    # Load SigLIP model if feature extraction is enabled
    siglip_model = None
    siglip_processor = None

    if agg_config['extract_features']:
        logger.info(f"Loading SigLIP model: {agg_config['siglip_model']}")
        from transformers import AutoModel, AutoProcessor

        siglip_processor = AutoProcessor.from_pretrained(
            agg_config['siglip_model'], use_fast=False
        )
        siglip_model = AutoModel.from_pretrained(
            agg_config['siglip_model']
        ).to(exp_config['device']).eval()

    # Auto-detect or use specified levels
    levels_to_use = exp_config.get('levels', None)
    if levels_to_use is None:
        from my3dis.aggregation.aggregation_pipeline import detect_available_levels
        exp_dir = Path(exp_config['exp_dir'])
        detected_levels = detect_available_levels(exp_dir)
        if detected_levels:
            levels_to_use = detected_levels
            logger.info(f"Auto-detected levels from experiment directory: {levels_to_use}")
        else:
            logger.warning("No levels detected and none specified, pipeline will use defaults")

    # Run aggregation
    start_time = time.time()
    results = pipeline.run(
        output_dir=exp_config['aggregation_output_dir'],
        levels=levels_to_use,
        extract_features=agg_config['extract_features'],
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
    )
    elapsed = time.time() - start_time

    logger.info(f"✓ Aggregation completed in {elapsed:.1f}s")
    logger.info(f"  Output: {exp_config['aggregation_output_dir']}")

    return results


def auto_detect_available_levels(aggregation_output_dir: Path) -> List[str]:
    """
    Auto-detect available levels from aggregation output directory.

    Scans for proposal_data_level*.npz files and returns available level names.

    Args:
        aggregation_output_dir: Path to aggregation output directory

    Returns:
        List of available level names (e.g., ['L2', 'L4', 'L6'])

    Example:
        If directory contains:
        - proposal_data_level2.npz
        - proposal_data_level4.npz
        - proposal_data_level6.npz

        Returns: ['L2', 'L4', 'L6']
    """
    import re

    if not aggregation_output_dir.exists():
        logger.warning(f"Aggregation output directory not found: {aggregation_output_dir}")
        return []

    # Find all proposal_data_level*.npz files
    proposal_files = list(aggregation_output_dir.glob("proposal_data_level*.npz"))

    # Extract level numbers
    levels = []
    for file in proposal_files:
        match = re.search(r'level(\d+)\.npz', file.name)
        if match:
            level_num = int(match.group(1))
            levels.append(f'L{level_num}')

    # Sort levels by number
    levels.sort(key=lambda x: int(x[1:]))

    if levels:
        logger.info(f"Auto-detected available levels: {levels}")
    else:
        logger.warning(f"No proposal data files found in {aggregation_output_dir}")

    return levels


def run_inference_stage(
    config: Dict,
    proposal_data_paths: Dict[str, str],
    strategy: Optional[str] = None,
    output_subdir: Optional[str] = None
) -> str:
    """
    Run inference stage: 3D proposals + features → OV-3DIS predictions.

    Args:
        config: Full configuration dictionary
        proposal_data_paths: Dictionary mapping level -> NPZ file path
        strategy: Strategy to use (overrides config if specified)
        output_subdir: Output subdirectory name (for multi-strategy testing)

    Returns:
        Path to predictions JSON file
    """
    logger.info("="*80)
    logger.info(f"STAGE 2: INFERENCE (OV-3DIS{f' - {strategy}' if strategy else ''})")
    logger.info("="*80)

    exp_config = config['experiment']
    scene_name = Path(exp_config['exp_dir']).name
    inf_config_dict = config['inference'].copy()

    # Override strategy if specified
    if strategy is not None:
        inf_config_dict['strategy'] = strategy
        logger.info(f"Using strategy: {strategy}")

    # Auto-detect available levels from aggregation output
    aggregation_output_dir = Path(exp_config.get('aggregation_output_dir',
                                                  Path(exp_config['exp_dir']) / 'aggregation_output'))
    available_levels = auto_detect_available_levels(aggregation_output_dir)

    # Auto-configure level settings if not explicitly set or set to 'auto'
    # For each strategy, check if levels are null/auto and fill with detected levels
    if available_levels:
        # Helper function to determine object and part levels from available levels
        def get_default_level_split(levels):
            """Split available levels into object (coarser) and part (finer) levels."""
            if len(levels) >= 2:
                # Use coarser levels for objects, finer levels for parts
                mid = len(levels) // 2
                object_levels = levels[:mid] if mid > 0 else levels[:1]
                part_levels = levels[mid:] if mid < len(levels) else levels[-1:]
                return object_levels, part_levels
            elif len(levels) == 1:
                # Only one level - use it for both
                return levels, levels
            else:
                return [], []

        object_levels_default, part_levels_default = get_default_level_split(available_levels)

        # Update retrieval config
        if 'retrieval' in inf_config_dict:
            if inf_config_dict['retrieval'].get('object_levels') in [None, 'auto', []]:
                inf_config_dict['retrieval']['object_levels'] = object_levels_default
                logger.info(f"Auto-configured retrieval.object_levels = {object_levels_default}")
            if inf_config_dict['retrieval'].get('part_levels') in [None, 'auto', []]:
                inf_config_dict['retrieval']['part_levels'] = part_levels_default
                logger.info(f"Auto-configured retrieval.part_levels = {part_levels_default}")

        # Update independent strategy config
        if 'independent' in inf_config_dict:
            if inf_config_dict['independent'].get('object_levels') in [None, 'auto', []]:
                inf_config_dict['independent']['object_levels'] = object_levels_default
                logger.info(f"Auto-configured independent.object_levels = {object_levels_default}")
            if inf_config_dict['independent'].get('part_levels') in [None, 'auto', []]:
                inf_config_dict['independent']['part_levels'] = part_levels_default
                logger.info(f"Auto-configured independent.part_levels = {part_levels_default}")

        # Update hierarchical strategy config (uses all levels for coarse-to-fine)
        # Hierarchical doesn't have object_levels/part_levels, skip

        # Update exhaustive_pairing strategy config
        if 'exhaustive_pairing' in inf_config_dict:
            if inf_config_dict['exhaustive_pairing'].get('available_levels') in [None, 'auto', []]:
                inf_config_dict['exhaustive_pairing']['available_levels'] = available_levels
                logger.info(f"Auto-configured exhaustive_pairing.available_levels = {available_levels}")

    # Convert inference config dict to InferenceConfig object (from dict, not file)
    inference_config = _dict_to_inference_config(inf_config_dict)

    # Load proposals and features
    logger.info("Loading proposals and features from aggregation output...")
    proposals_by_level = {}
    scene_num_points = None

    for level_str, npz_path in proposal_data_paths.items():
        level = int(level_str)
        level_key = f"L{level}"

        import numpy as np
        data = np.load(npz_path, allow_pickle=False)

        proposal_masks = torch.from_numpy(data['proposal_masks'])
        proposal_features = torch.from_numpy(data['proposal_features'])
        proposal_ids = data['proposal_ids']
        if scene_num_points is None:
            scene_num_points = int(proposal_masks.shape[0])

        logger.info(f"  Level {level}: {len(proposal_ids)} proposals")

        # Import Proposal class
        from my3dis.inference.retrieval import Proposal

        # Create Proposal objects
        proposals = []
        for i, prop_id in enumerate(proposal_ids):
            proposals.append(Proposal(
                proposal_id=int(prop_id),
                level=level_key,
                mask=proposal_masks[:, i].numpy(),
                feature=proposal_features[i].numpy(),
                object_id=int(prop_id),  # Same as proposal_id for tracking outputs
            ))

        proposals_by_level[level_key] = proposals

    # Check for family tree
    logger.info("Checking for family tree...")
    family_tree_path = Path(exp_config['exp_dir']) / 'relations' / 'family_tree.json'

    if not family_tree_path.exists():
        logger.warning(f"Family tree not found: {family_tree_path}")
        family_tree_path_str = None
    else:
        family_tree_path_str = str(family_tree_path)
        logger.info(f"Found family tree: {family_tree_path_str}")

    strategy = inference_config.strategy

    # Determine SigLIP/CLIP weights for inference + combined queries
    if strategy in config['inference'] and 'siglip_model' in config['inference'][strategy]:
        text_model = config['inference'][strategy]['siglip_model']
        logger.info(f"Loading SigLIP model for text encoding (from {strategy} strategy config): {text_model}")
    else:
        text_model = config['aggregation']['siglip_model']
        logger.info(f"Loading SigLIP model for text encoding (from aggregation config): {text_model}")

    # 使用优化版本的 feature extractor（可通过全局开关控制）
    if USE_OPTIMIZED_EXTRACTION:
        logger.info(f"Using OPTIMIZED SigLIP feature extractor (batch_size=32, cache_size=100)")
        feature_extractor = OptimizedSigLIPFeatureExtractor(
            model_name=text_model,
            device=exp_config['device'],
            batch_size=32,  # 批处理大小（可根据 GPU 显存调整）
            cache_size=100   # 图片缓存大小
        )
    else:
        logger.info(f"Using original SigLIP feature extractor")
        feature_extractor = SigLIPFeatureExtractor(
            text_model,
            exp_config['device']
        )

    # Initialize inference pipeline
    logger.info(f"Initializing inference pipeline (strategy: {strategy})")

    # Extract strategy-specific kwargs
    strategy_kwargs = {}
    if strategy == 'independent':
        ind_cfg = inference_config.independent
        strategy_kwargs = {
            'object_levels': ind_cfg.object_levels,
            'part_levels': ind_cfg.part_levels,
            'top_k_per_level': ind_cfg.top_k_per_level,
            'retrieval_threshold': ind_cfg.retrieval_threshold,
            'min_retrieval_threshold': ind_cfg.min_retrieval_threshold,
            'threshold_backoff_step': ind_cfg.threshold_backoff_step,
            'fallback_to_topk': ind_cfg.fallback_to_topk,
            'use_combined_query': ind_cfg.use_combined_query,
            'siglip_model': ind_cfg.siglip_model,
            'siglip_device': ind_cfg.siglip_device,
            'scale_semantic_score': ind_cfg.scale_semantic_score,
            'apply_softmax': ind_cfg.apply_softmax,
            'feature_extractor': feature_extractor,
        }
    elif strategy == 'hierarchical':
        hier_cfg = inference_config.hierarchical
        strategy_kwargs = {
            'coarse_top_k': hier_cfg.coarse_top_k,
            'object_top_k': hier_cfg.object_top_k,
            'part_top_k': hier_cfg.part_top_k,
            'coarse_threshold': hier_cfg.coarse_threshold,
            'refinement_threshold': hier_cfg.refinement_threshold,
            'refinement_margin': hier_cfg.refinement_margin,
            'use_combined_query': hier_cfg.use_combined_query,
            'siglip_model': hier_cfg.siglip_model,
            'siglip_device': hier_cfg.siglip_device,
            'scale_semantic_score': hier_cfg.scale_semantic_score,
            'apply_softmax': hier_cfg.apply_softmax,
            'feature_extractor': feature_extractor,
        }
    elif strategy == 'exhaustive_pairing':
        exh_cfg = inference_config.exhaustive_pairing
        strategy_kwargs = {
            'available_levels': exh_cfg.available_levels,
            'top_k_per_level': exh_cfg.top_k_per_level,
            'retrieval_threshold': exh_cfg.retrieval_threshold,
            'min_retrieval_threshold': exh_cfg.min_retrieval_threshold,
            'threshold_backoff_step': exh_cfg.threshold_backoff_step,
            'fallback_to_topk': exh_cfg.fallback_to_topk,
            'selection_metric': exh_cfg.selection_metric,
            'use_tree_pruning': exh_cfg.use_tree_pruning,
            'return_all_pairs': exh_cfg.return_all_pairs,
            'use_combined_query': exh_cfg.use_combined_query,
        }

        if exh_cfg.use_combined_query:
            strategy_kwargs.update({
                'siglip_model': exh_cfg.siglip_model,
                'siglip_device': exh_cfg.siglip_device,
                'scale_semantic_score': exh_cfg.scale_semantic_score,
                'apply_softmax': exh_cfg.apply_softmax,
                'feature_extractor': feature_extractor,
            })

    pipeline = InferencePipeline(
        proposals_by_level=proposals_by_level,
        family_tree_path=family_tree_path_str,  # Pass path instead of dict
        strategy=strategy,
        config=inference_config,  # Pass config for scoring/pairing/nms
        **strategy_kwargs
    )
    if scene_num_points is None:
        scene_num_points = pipeline.num_points
    elif pipeline.num_points is not None:
        scene_num_points = pipeline.num_points

    # Load Search3D queries
    logger.info("Loading Search3D queries...")
    try:
        # Add eval directory to path for imports
        eval_dir = Path(__file__).parent.parent / 'eval' / 'search3d'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))

        from data.multiscan_search3d_constants import (
            VALID_JOINT_TUPLE_NAMES,
            VALID_JOINT_SEMANTIC_IDS
        )

        # Create queries from Search3D constants
        # Each query: (object_text, part_text, label_id, num_instances)
        queries = []
        for (obj_text, part_text), label_id in zip(VALID_JOINT_TUPLE_NAMES, VALID_JOINT_SEMANTIC_IDS):
            # For now, assume single instance per class
            # TODO: Load actual instance counts from GT if needed
            queries.append((obj_text, part_text, label_id, 1))

        logger.info(f"Loaded {len(queries)} Search3D queries")
        logger.info(f"Sample queries: {queries[:5]}")

        # Optional: filter queries to labels present in GT for this scene
        eval_cfg = config.get('evaluation', {})
        # Try new gt_paths.obj_part first, fallback to deprecated gt_path
        gt_path = None
        if 'gt_paths' in eval_cfg and eval_cfg['gt_paths']:
            gt_path = eval_cfg['gt_paths'].get('obj_part')
        if not gt_path:
            gt_path = eval_cfg.get('gt_path')

        scene_label_counts = load_scene_label_distribution(gt_path, scene_name)
        if scene_label_counts:
            allowed_labels = set(scene_label_counts.keys())
            filtered_queries = [q for q in queries if q[2] in allowed_labels]
            if filtered_queries:
                logger.info(
                    "Scene %s: restricting queries to %d/%d GT label_ids",
                    scene_name, len(filtered_queries), len(queries)
                )
                sample_labels = sorted(allowed_labels)[:10]
                logger.info("  GT label ids (sample): %s", sample_labels)
                queries = filtered_queries
            else:
                logger.warning(
                    "GT label filtering removed all queries for scene %s; keeping original list",
                    scene_name
                )

    except Exception as e:
        logger.warning(f"Failed to load Search3D queries: {e}")
        logger.warning("Falling back to demo queries...")
        # Fallback to valid Search3D queries (not demo values!)
        queries = [
            ("door", "frame", 4008, 1),         # door + frame
            ("cabinet", "door", 11002, 1),      # cabinet + door
            ("toilet", "lid", 32005, 1),        # toilet + lid
            ("toilet", "tank_cover", 32007, 1), # toilet + tank_cover
            ("chair", "base", 8010, 1),         # chair + base
        ]

    logger.info(f"Running inference on {len(queries)} queries...")

    # Setup output directory (with optional subdirectory for multi-strategy testing)
    base_output_dir = Path(exp_config['inference_output_dir'])
    if output_subdir:
        output_dir = base_output_dir / output_subdir
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    predictions = []
    prediction_masks = []  # Store masks for evaluation
    prediction_results = []  # Store full Prediction objects for Search3D export
    start_time = time.time()

    for object_text, part_text, label_id, num_instances in queries:
        logger.info(f"  Query: object='{object_text}', part='{part_text}' (label={label_id}, instances={num_instances})")

        # Encode text queries
        object_features = feature_extractor.extract_text_features([object_text])
        part_features = feature_extractor.extract_text_features([part_text])

        object_feat = object_features[0]
        part_feat = part_features[0]

        # Run inference
        results = pipeline.infer_single_query(
            object_query_feat=object_feat,
            part_query_feat=part_feat,
            label_id=label_id,
        )

        # Store predictions
        for i, result in enumerate(results):
            pred_dict = {
                'prediction_id': len(predictions),  # Unique ID for this prediction
                'object_query': object_text,
                'part_query': part_text,
                'label_id': label_id,
                'num_instances': num_instances,
                'score': float(result.score),
                'mask_points': int(result.mask.sum()),
            }

            # Add metadata if available
            if result.metadata:
                pred_dict.update({
                    'object_level': result.metadata.get('object_level'),
                    'part_level': result.metadata.get('part_level'),
                    'object_id': result.metadata.get('object_id'),
                    'part_id': result.metadata.get('part_id'),
                })

            predictions.append(pred_dict)
            prediction_masks.append(result.mask)  # Store full mask
            prediction_results.append(result)  # Store full result for metadata access

        logger.info(f"    Found {len(results)} predictions")

    elapsed = time.time() - start_time

    # Save predictions JSON
    predictions_path = output_dir / 'predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Save masks NPZ for evaluation
    if len(prediction_masks) > 0:
        import numpy as np
        masks_array = np.column_stack([m for m in prediction_masks])  # Shape: (N_points, N_predictions)
        masks_bool = masks_array.astype(bool)
        masks_path = output_dir / 'prediction_masks.npz'
        np.savez_compressed(
            masks_path,
            masks=masks_bool,
            num_predictions=len(predictions)
        )
        logger.info(f"  Saved prediction masks: {masks_path}")

        if config['inference'].get('export_predictions_ply', False):
            try:
                points, colors = load_scene_pointcloud(exp_config['scene_path'])
                export_dir = output_dir / 'ply_exports'
                max_exports = int(config['inference'].get('ply_export_max_predictions', 20))
                palette_cfg = config['inference'].get('ply_export_colors', None)
                include_full_scene = bool(
                    config['inference'].get('ply_export_include_full_scene', True)
                )
                background_cfg = config['inference'].get('ply_export_background_color', [80, 80, 80])
                if background_cfg is None:
                    background_color = None
                elif (
                    isinstance(background_cfg, (list, tuple))
                    and len(background_cfg) == 3
                ):
                    background_color = tuple(int(c) for c in background_cfg)
                else:
                    logger.warning(
                        "Invalid ply_export_background_color=%s; falling back to gray", background_cfg
                    )
                    background_color = (80, 80, 80)
                default_palette = [
                    (255, 99, 71),
                    (102, 205, 170),
                    (65, 105, 225),
                    (218, 112, 214),
                    (255, 215, 0),
                    (0, 191, 255),
                    (144, 238, 144),
                    (255, 140, 0),
                ]
                if palette_cfg:
                    palette = [
                        tuple(int(c) for c in color)
                        for color in palette_cfg
                        if isinstance(color, (list, tuple)) and len(color) == 3
                    ]
                    if not palette:
                        palette = default_palette
                else:
                    palette = default_palette

                exported = 0
                limit = min(len(predictions), max_exports)
                for idx in range(limit):
                    mask = masks_bool[:, idx]
                    pred = predictions[idx]
                    color = palette[idx % len(palette)]
                    ply_path = export_dir / f"pred_{idx:04d}_label_{pred['label_id']}.ply"
                    if save_mask_as_ply(
                        points,
                        colors,
                        mask,
                        ply_path,
                        highlight_color=color,
                        include_full_scene=include_full_scene,
                        background_color=background_color,
                    ):
                        exported += 1

                logger.info("  Exported %d/%d prediction PLY files to %s", exported, limit, export_dir)
            except Exception as e:
                logger.warning("Failed to export predictions as PLY: %s", e, exc_info=True)
    else:
        masks_path = None

    # Export Search3D format (obj_part only)
    search3d_export_stats = None
    if config['evaluation'].get('enabled', False):
        try:
            from my3dis.aggregation.search3d_formatter import (
                export_predictions_search3d,
                load_point_cloud_for_export
            )

            logger.info("Exporting predictions in Search3D format...")

            # Load point cloud to get total number of points
            # Priority: GT PLY path > scene path
            scene_name = Path(exp_config['scene_path']).name
            gt_ply_path = config['evaluation'].get('gt_ply_path')

            points = None
            num_points = None

            # When no predictions exist, fall back to known point count from proposals
            if len(prediction_masks) == 0 and scene_num_points is not None:
                num_points = scene_num_points

            # Try GT PLY path first (for Search3D evaluation)
            if gt_ply_path:
                gt_ply_file = Path(gt_ply_path) / f"{scene_name}.ply"
                if gt_ply_file.exists():
                    try:
                        logger.info(f"  Loading GT PLY from {gt_ply_file}")
                        points = load_point_cloud_for_export(
                            gt_ply_file.parent,
                            mesh_name=gt_ply_file.name
                        )
                        num_points = len(points)
                        logger.info(f"  Scene has {num_points} points (from GT PLY)")
                    except Exception as e:
                        logger.warning(f"Failed to load GT PLY: {e}")

            # Fall back to scene path if GT PLY not available
            if num_points is None:
                scene_path = Path(exp_config['scene_path'])
                try:
                    points = load_point_cloud_for_export(scene_path)
                    num_points = len(points)
                    logger.info(f"  Scene has {num_points} points (from scene path)")
                except Exception as e:
                    logger.warning(f"Failed to load point cloud: {e}")
                    logger.warning("Using mask dimensions to infer number of points")
                    if len(prediction_masks) > 0:
                        num_points = len(prediction_masks[0])
                    elif scene_num_points is not None:
                        num_points = scene_num_points
                    else:
                        raise RuntimeError(
                            "Unable to determine number of points for Search3D export"
                        )

            # Create Prediction-like objects for formatter
            class SimplePrediction:
                def __init__(self, mask, label_id, score, metadata=None):
                    self.mask = mask
                    self.label_id = label_id
                    self.score = score
                    self.metadata = metadata or {}

            search3d_export_stats = {}

            logger.info("Exporting part-level predictions...")

            # Create predictions with part masks
            part_pred_list = []
            for i, (result, pred_dict) in enumerate(zip(prediction_results, predictions)):
                # Use part_mask from metadata if available
                if result.metadata and 'part_mask' in result.metadata:
                    part_mask = result.metadata['part_mask']
                else:
                    # Fallback to main mask
                    part_mask = result.mask

                part_pred_list.append(SimplePrediction(
                    mask=part_mask,
                    label_id=pred_dict['label_id'],  # Use composite ID
                    score=pred_dict['score'],
                    metadata=pred_dict
                ))

            part_stats = export_predictions_search3d(
                predictions=part_pred_list,
                num_points=num_points,
                output_path=output_dir / f"{scene_name}_obj_part_inst.txt",
                format_type='obj_part',
            )
            search3d_export_stats['obj_part'] = part_stats
            logger.info(f"  ✓ Part-level: {part_stats['output_path']}")

            logger.info("✓ Search3D format export completed")

        except Exception as e:
            logger.warning(f"Failed to export Search3D format: {e}", exc_info=True)
            search3d_export_stats = None

    logger.info(f"✓ Inference completed in {elapsed:.1f}s")
    logger.info(f"  Output: {predictions_path}")
    logger.info(f"  Total predictions: {len(predictions)}")

    # Return statistics for reporting
    return {
        'predictions_path': str(predictions_path),
        'masks_path': str(masks_path) if masks_path else None,
        'num_predictions': len(predictions),
        'elapsed_time': elapsed,
        'strategy': inf_config_dict['strategy'],
        'output_dir': str(output_dir),
        'search3d_export': search3d_export_stats,
    }


def generate_strategy_report(
    strategy_name: str,
    inference_stats: Dict,
    config: Dict,
    output_path: str,
    eval_results: Optional[Dict] = None
):
    """
    Generate evaluation report for a single strategy.

    Args:
        strategy_name: Name of the strategy (e.g., "independent", "hierarchical")
        inference_stats: Statistics from run_inference_stage
        config: Full configuration dictionary
        output_path: Path to save the report
        eval_results: Optional evaluation results from run_evaluation_stage
    """
    logger.info(f"Generating report for strategy: {strategy_name}")

    report_lines = [
        f"# Inference Strategy Report: {strategy_name.upper()}",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration Summary",
        "",
        f"- **Strategy**: {strategy_name}",
        f"- **Experiment Dir**: {config['experiment']['exp_dir']}",
        f"- **Scene Path**: {config['experiment']['scene_path']}",
        f"- **Device**: {config['experiment']['device']}",
        "",
        "## Inference Settings",
        "",
    ]

    # Add strategy-specific settings
    inf_config = config['inference']
    if strategy_name == 'independent':
        ind_cfg = inf_config.get('independent', {})
        report_lines.extend([
            f"- **Object Levels**: {ind_cfg.get('object_levels', [])}",
            f"- **Part Levels**: {ind_cfg.get('part_levels', [])}",
            f"- **Top-K per Level**: {ind_cfg.get('top_k_per_level', 0)}",
            f"- **Retrieval Threshold**: {ind_cfg.get('retrieval_threshold', 0)}",
            f"- **Min Retrieval Threshold**: {ind_cfg.get('min_retrieval_threshold', 0)}",
        ])
    elif strategy_name == 'hierarchical':
        hier_cfg = inf_config.get('hierarchical', {})
        report_lines.extend([
            f"- **Coarse Top-K**: {hier_cfg.get('coarse_top_k', 0)}",
            f"- **Object Top-K**: {hier_cfg.get('object_top_k', 0)}",
            f"- **Part Top-K**: {hier_cfg.get('part_top_k', 0)}",
            f"- **Coarse Threshold**: {hier_cfg.get('coarse_threshold', 0)}",
            f"- **Refinement Threshold**: {hier_cfg.get('refinement_threshold', 0)}",
        ])

    # Add common settings
    report_lines.extend([
        "",
        "### Retrieval Settings",
        f"- **Temperature**: {inf_config['retrieval']['temperature']}",
        f"- **Min Similarity**: {inf_config['retrieval']['min_similarity']}",
        "",
        "### NMS Settings",
        f"- **IoU Threshold**: {inf_config['nms']['iou_threshold']}",
        f"- **Keep Top-K**: {inf_config['nms']['keep_top_k']}",
        f"- **Use Soft NMS**: {inf_config['nms']['use_soft_nms']}",
        "",
        "### Scoring Settings",
        f"- **Method**: {inf_config['scoring']['method']}",
        f"- **Weights**: object={inf_config['scoring']['weights']['object']}, part={inf_config['scoring']['weights']['part']}, geometric={inf_config['scoring']['weights']['geometric']}",
        "",
        "## Results",
        "",
        f"- **Total Predictions**: {inference_stats['num_predictions']}",
        f"- **Inference Time**: {inference_stats['elapsed_time']:.2f}s",
        f"- **Output Directory**: {inference_stats['output_dir']}",
        f"- **Predictions File**: {inference_stats['predictions_path']}",
        "",
        "## Evaluation Metrics",
        "",
    ])

    # Add evaluation results if available
    if eval_results and eval_results.get('status') == 'success':
        report_lines.extend([
            f"- **mAP**: {eval_results['mAP']:.3f}",
            f"- **AP@50**: {eval_results['AP50']:.3f}",
            f"- **AP@25**: {eval_results['AP25']:.3f}",
            "",
            "### Per-Class Results",
            "",
            "| Class | AP | AP@50 | AP@25 |",
            "|-------|-----|-------|-------|",
        ])

        # Add per-class results if available
        if 'metrics' in eval_results and 'classes' in eval_results['metrics']:
            for class_name, class_metrics in eval_results['metrics']['classes'].items():
                ap = class_metrics.get('ap', 0)
                ap50 = class_metrics.get('ap50%', 0)
                ap25 = class_metrics.get('ap25%', 0)
                report_lines.append(
                    f"| {class_name} | {ap:.3f} | {ap50:.3f} | {ap25:.3f} |"
                )
    else:
        if eval_results and eval_results.get('status') == 'failed':
            report_lines.append(f"*Evaluation failed: {eval_results.get('error', 'unknown error')}*")
        else:
            report_lines.append("*Evaluation metrics not available or evaluation was skipped.*")

    report_lines.extend([
        "",
        "---",
        "",
        f"Report generated by My3DIS full pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"  Report saved: {output_path}")


def generate_comparison_report(
    strategy_results: Dict[str, Dict],
    config: Dict,
    output_path: str
):
    """
    Generate comparison report for all tested strategies.

    Args:
        strategy_results: Dictionary mapping strategy_name -> inference_stats
        config: Full configuration dictionary
        output_path: Path to save the comparison report
    """
    logger.info("Generating strategy comparison report...")

    report_lines = [
        "# Multi-Strategy Inference Comparison Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Overview",
        "",
        f"- **Experiment Dir**: {config['experiment']['exp_dir']}",
        f"- **Scene Path**: {config['experiment']['scene_path']}",
        f"- **Device**: {config['experiment']['device']}",
        f"- **Levels**: {config['experiment']['levels']}",
        "",
        "## Strategy Results Summary",
        "",
        "| Strategy | Predictions | Time (s) | Output Dir |",
        "|----------|-------------|----------|------------|",
    ]

    # Add rows for each strategy
    for strategy_name in sorted(strategy_results.keys()):
        stats = strategy_results[strategy_name]
        report_lines.append(
            f"| **{strategy_name}** | {stats['num_predictions']} | "
            f"{stats['elapsed_time']:.2f} | `{Path(stats['output_dir']).name}` |"
        )

    report_lines.extend([
        "",
        "## Detailed Comparison",
        "",
    ])

    # Add detailed section for each strategy
    for strategy_name in sorted(strategy_results.keys()):
        stats = strategy_results[strategy_name]
        report_lines.extend([
            f"### {strategy_name.upper()} Strategy",
            "",
            f"- **Predictions**: {stats['num_predictions']}",
            f"- **Inference Time**: {stats['elapsed_time']:.2f}s",
            f"- **Predictions per Second**: {stats['num_predictions'] / stats['elapsed_time']:.2f}",
            f"- **Output Directory**: `{stats['output_dir']}`",
            f"- **Predictions File**: `{stats['predictions_path']}`",
            f"- **Individual Report**: [{strategy_name}_report.md](./{Path(stats['output_dir']).name}/{strategy_name}_report.md)",
            "",
        ])

    # Add performance comparison
    if len(strategy_results) > 1:
        fastest = min(strategy_results.items(), key=lambda x: x[1]['elapsed_time'])
        most_predictions = max(strategy_results.items(), key=lambda x: x[1]['num_predictions'])

        report_lines.extend([
            "## Performance Highlights",
            "",
            f"- **Fastest Strategy**: {fastest[0]} ({fastest[1]['elapsed_time']:.2f}s)",
            f"- **Most Predictions**: {most_predictions[0]} ({most_predictions[1]['num_predictions']} predictions)",
            "",
        ])

    report_lines.extend([
        "## Evaluation Metrics Comparison",
        "",
        "*Evaluation metrics comparison will be added when evaluation stage is implemented.*",
        "",
        "---",
        "",
        f"Report generated by My3DIS full pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"  Comparison report saved: {output_path}")


def convert_predictions_to_eval_format(
    predictions_path: str,
    masks_path: str,
    num_points: int
) -> Dict:
    """
    Convert predictions to Search3D evaluation format.

    Args:
        predictions_path: Path to predictions JSON file
        masks_path: Path to prediction_masks.npz file
        num_points: Number of points in the scene

    Returns:
        Dictionary with pred_classes, pred_scores, pred_masks
    """
    import numpy as np

    # Load predictions JSON
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        logger.warning(f"No predictions found in {predictions_path}")
        return {
            'pred_classes': np.array([], dtype=np.int32),
            'pred_scores': np.array([], dtype=np.float32),
            'pred_masks': np.zeros((num_points, 0), dtype=bool)
        }

    # Load masks NPZ
    masks_data = np.load(masks_path)
    pred_masks = masks_data['masks']  # Shape: (N_points, N_predictions)

    # Extract classes and scores
    pred_classes = []
    pred_scores = []

    for pred in predictions:
        label_id = pred.get('label_id', -1)
        score = pred.get('score', 0.0)
        pred_classes.append(label_id)
        pred_scores.append(score)

    return {
        'pred_classes': np.array(pred_classes, dtype=np.int32),
        'pred_scores': np.array(pred_scores, dtype=np.float32),
        'pred_masks': pred_masks
    }


def run_evaluation_stage(
    config: Dict,
    inference_output_dir: str,
    strategy: Optional[str] = None,
) -> Dict:
    """
    Run evaluation stage: Compute mAP for OV part-level predictions using the Search3D protocol.

    Args:
        config: Full configuration dictionary
        inference_output_dir: Directory containing inference outputs
            Expected files:
            - predictions.json
            - prediction_masks.npz (old format, for backward compat)
            - scene_*_obj_part_inst.txt (Search3D part-level format)
        strategy: Strategy name (for logging purposes)

    Returns:
        Dictionary with evaluation metrics:
        {
            'obj_part': {...},    # Part-level metrics
            'summary': {...}      # Combined summary
        }
    """
    strategy_suffix = f" ({strategy})" if strategy else ""
    logger.info("="*80)
    logger.info(f"STAGE 3: EVALUATION (Search3D OV-Part){strategy_suffix}")
    logger.info("="*80)

    eval_config = config['evaluation']
    exp_dir = Path(config['experiment']['exp_dir'])
    scene_name = exp_dir.name  # e.g., scene_00005_00
    inference_output_dir = Path(inference_output_dir)

    # Get evaluation modes from config (object evaluation removed)
    configured_modes = eval_config.get('eval_modes', ['obj_part'])
    if not isinstance(configured_modes, list):
        configured_modes = [configured_modes]

    unsupported_modes = [mode for mode in configured_modes if mode != 'obj_part']
    if unsupported_modes:
        logger.warning(
            "Skipping unsupported evaluation modes (object-level evaluation removed): %s",
            unsupported_modes
        )

    eval_modes = ['obj_part']

    logger.info(f"Scene: {scene_name}")
    logger.info(f"Evaluation modes: {eval_modes}")

    # Check for backward compatibility with old config format
    gt_paths = eval_config.get('gt_paths', {})
    if not gt_paths and eval_config.get('gt_path'):
        # Fall back to old single GT path (treat as obj_part)
        logger.warning("Using deprecated 'gt_path' config. Please migrate to 'gt_paths'.")
        gt_paths = {'obj_part': eval_config['gt_path']}

    if not gt_paths:
        logger.warning("No ground truth paths specified, skipping evaluation")
        return {'status': 'skipped', 'reason': 'no_gt_paths'}

    # Results for all modes
    results = {}

    # Evaluate each mode
    for mode in eval_modes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {mode.upper()}")
        logger.info(f"{'='*80}")

        # Check if GT path exists for this mode
        if mode not in gt_paths:
            logger.warning(f"GT path not configured for mode '{mode}', skipping")
            results[mode] = {'status': 'skipped', 'reason': 'no_gt_path'}
            continue

        gt_dir = Path(gt_paths[mode])
        if not gt_dir.exists():
            logger.warning(f"GT directory not found: {gt_dir}")
            results[mode] = {'status': 'skipped', 'reason': 'gt_dir_not_found'}
            continue

        # Determine GT filename based on mode (obj_part only)
        if mode == 'obj_part':
            gt_filename = f"{scene_name}_obj_part_inst.txt"
            pred_filename = f"{scene_name}_obj_part_inst.txt"
        else:
            logger.warning(f"Unknown evaluation mode: {mode}")
            results[mode] = {'status': 'skipped', 'reason': 'unknown_mode'}
            continue

        gt_file = gt_dir / gt_filename
        pred_file = inference_output_dir / pred_filename

        # Check files exist
        if not gt_file.exists():
            logger.warning(f"GT file not found: {gt_file}")
            results[mode] = {'status': 'skipped', 'reason': 'gt_file_not_found'}
            continue

        if not pred_file.exists():
            logger.warning(f"Prediction file not found: {pred_file}")
            results[mode] = {'status': 'skipped', 'reason': 'pred_file_not_found'}
            continue

        logger.info(f"  GT file: {gt_file}")
        logger.info(f"  Pred file: {pred_file}")

        try:
            # Get evaluation script path
            eval_scripts = eval_config.get('eval_scripts', {})
            if mode not in eval_scripts:
                logger.warning(f"Evaluation script not configured for mode '{mode}'")
                results[mode] = {'status': 'skipped', 'reason': 'no_eval_script'}
                continue

            eval_script_rel = eval_scripts[mode]
            project_root = Path(__file__).parent.parent
            eval_script = project_root / eval_script_rel

            if not eval_script.exists():
                logger.warning(f"Evaluation script not found: {eval_script}")
                results[mode] = {'status': 'skipped', 'reason': 'eval_script_not_found'}
                continue

            # Add eval directory to sys.path
            eval_dir = eval_script.parent
            if str(eval_dir) not in sys.path:
                sys.path.insert(0, str(eval_dir))

            # Import evaluation functions (obj_part only)
            if mode == 'obj_part':
                from eval_semantic_instance_parts_OV import (
                    evaluate_scan_from_files,
                    compute_averages
                )

            # Run evaluation
            logger.info(f"Running evaluation for {mode}...")
            matches = evaluate_scan_from_files(
                pred_file=str(pred_file),
                gt_file=str(gt_file),
                scene_name=scene_name
            )

            # Compute metrics
            from eval_semantic_instance_parts_OV import evaluate_matches
            ap = evaluate_matches({scene_name: matches})
            avgs = compute_averages(ap)

            logger.info(f"✓ {mode.upper()} evaluation completed{strategy_suffix}")
            logger.info(f"  mAP: {avgs['all_ap']:.3f}")
            logger.info(f"  AP@50: {avgs['all_ap_50%']:.3f}")
            logger.info(f"  AP@25: {avgs['all_ap_25%']:.3f}")

            results[mode] = {
                'status': 'success',
                'metrics': avgs,
                'scene': scene_name,
                'mAP': avgs['all_ap'],
                'AP50': avgs['all_ap_50%'],
                'AP25': avgs['all_ap_25%'],
            }

        except Exception as e:
            logger.error(f"{mode.upper()} evaluation failed: {e}", exc_info=True)
            results[mode] = {
                'status': 'failed',
                'error': str(e)
            }

    # Generate summary
    summary = {
        'total_modes': len(eval_modes),
        'successful': sum(1 for r in results.values() if r.get('status') == 'success'),
        'failed': sum(1 for r in results.values() if r.get('status') == 'failed'),
        'skipped': sum(1 for r in results.values() if r.get('status') == 'skipped'),
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  Total modes: {summary['total_modes']}")
    logger.info(f"  Successful: {summary['successful']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Skipped: {summary['skipped']}")

    # Print detailed results
    for mode, result in results.items():
        if result.get('status') == 'success':
            logger.info(f"\n  {mode.upper()}{strategy_suffix}:")
            logger.info(f"    mAP: {result['mAP']:.3f}")
            logger.info(f"    AP@50: {result['AP50']:.3f}")
            logger.info(f"    AP@25: {result['AP25']:.3f}")

    results['summary'] = summary
    return results


def discover_scenes(exp_root: Path, scenes_config) -> List[Tuple[str, Path, Path]]:
    """
    Discover scenes to process in multi-scene mode.

    Args:
        exp_root: Experiment root directory
        scenes_config: Scene configuration from YAML (all, list, or pattern)

    Returns:
        List of (scene_name, scene_exp_dir, scene_data_path) tuples
    """
    import glob as glob_module

    scene_dirs = []

    # Find all scene directories in exp_root
    for item in exp_root.iterdir():
        if item.is_dir() and item.name.startswith('scene_'):
            scene_dirs.append(item)

    scene_dirs.sort()

    if not scene_dirs:
        logger.warning(f"No scene directories found in {exp_root}")
        return []

    logger.info(f"Found {len(scene_dirs)} scene directories in {exp_root}")

    # Filter based on scenes_config
    if scenes_config is None or scenes_config == "all":
        # Process all scenes
        selected_scenes = scene_dirs
    elif isinstance(scenes_config, list):
        # Process specific scenes
        selected_names = set(scenes_config)
        selected_scenes = [d for d in scene_dirs if d.name in selected_names]
    elif isinstance(scenes_config, str):
        # Pattern matching
        import fnmatch
        pattern = scenes_config
        selected_scenes = [d for d in scene_dirs if fnmatch.fnmatch(d.name, pattern)]
    else:
        logger.error(f"Invalid scenes config: {scenes_config}")
        return []

    logger.info(f"Selected {len(selected_scenes)} scenes to process")

    return [(d.name, d, None) for d in selected_scenes]


def process_single_scene(
    scene_name: str,
    scene_exp_dir: Path,
    scene_data_path: Optional[Path],
    config: Dict,
    dataset_root: Optional[Path] = None
) -> Dict:
    """
    Process a single scene through the full pipeline.

    Args:
        scene_name: Scene name (e.g., scene_00005_00)
        scene_exp_dir: Scene experiment directory
        scene_data_path: Scene data path (PLY, images, etc.)
        config: Full configuration dictionary
        dataset_root: Dataset root directory (for auto-resolving scene_data_path)

    Returns:
        Dictionary with scene results and metrics
    """
    logger.info("="*80)
    logger.info(f"PROCESSING SCENE: {scene_name}")
    logger.info("="*80)

    # Resolve scene_data_path if not provided
    if scene_data_path is None and dataset_root is not None:
        scene_data_path = dataset_root / scene_name
        logger.info(f"Auto-resolved scene_data_path: {scene_data_path}")

    if scene_data_path is None or not scene_data_path.exists():
        logger.error(f"Scene data path not found: {scene_data_path}")
        return {
            'status': 'failed',
            'error': 'scene_data_path_not_found',
            'scene_name': scene_name
        }

    # Create scene-specific config
    scene_config = config.copy()
    scene_config['experiment'] = config['experiment'].copy()
    scene_config['experiment']['exp_dir'] = str(scene_exp_dir)
    scene_config['experiment']['scene_path'] = str(scene_data_path)

    # Reset output directories for this scene (to trigger model-based naming)
    scene_config['experiment']['aggregation_output_dir'] = None
    scene_config['experiment']['inference_output_dir'] = None

    # Setup paths for this scene (will use model-based naming if _model_info exists)
    scene_config = setup_paths(scene_config)

    scene_results = {
        'scene_name': scene_name,
        'exp_dir': str(scene_exp_dir),
        'scene_path': str(scene_data_path),
        'status': 'success',
        'aggregation': {},
        'inference': {},
        'evaluation': {}
    }

    try:
        # Stage 1: Aggregation
        proposal_data_paths = None
        if scene_config['aggregation']['enabled']:
            try:
                proposal_data_paths = run_aggregation_stage(scene_config)
                scene_results['aggregation'] = {
                    'status': 'success',
                    'outputs': proposal_data_paths
                }
            except Exception as e:
                logger.error(f"Aggregation failed for {scene_name}: {e}", exc_info=True)
                scene_results['aggregation'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                scene_results['status'] = 'failed'
                return scene_results
        else:
            # Load existing aggregation outputs
            logger.info(f"Skipping aggregation for {scene_name} (loading existing outputs)")
            agg_output_dir = Path(scene_config['experiment']['aggregation_output_dir'])
            proposal_data_paths = {}

            levels_to_check = scene_config['experiment'].get('levels', None)
            if levels_to_check is None:
                from my3dis.aggregation.aggregation_pipeline import detect_available_levels
                detected_levels = detect_available_levels(scene_exp_dir)
                if detected_levels:
                    levels_to_check = detected_levels
                else:
                    levels_to_check = [2, 4, 6]

            for level in levels_to_check:
                npz_path = agg_output_dir / f"proposal_data_level{level}.npz"
                if npz_path.exists():
                    proposal_data_paths[str(level)] = str(npz_path)

            scene_results['aggregation'] = {
                'status': 'skipped',
                'outputs': proposal_data_paths
            }

        # Stage 2: Inference
        strategy_results = {}
        if scene_config['inference']['enabled'] and proposal_data_paths:
            try:
                test_all_strategies = scene_config['inference'].get('test_all_strategies', False)

                if test_all_strategies:
                    strategies_to_test = scene_config['inference'].get('strategies_to_test', ['independent', 'hierarchical'])
                    for strategy in strategies_to_test:
                        inference_stats = run_inference_stage(
                            scene_config,
                            proposal_data_paths,
                            strategy=strategy,
                            output_subdir=strategy
                        )
                        strategy_results[strategy] = inference_stats
                else:
                    strategy_name = scene_config['inference']['strategy']
                    inference_stats = run_inference_stage(scene_config, proposal_data_paths)
                    strategy_results[strategy_name] = inference_stats

                scene_results['inference'] = {
                    'status': 'success',
                    'strategies': strategy_results
                }
            except Exception as e:
                logger.error(f"Inference failed for {scene_name}: {e}", exc_info=True)
                scene_results['inference'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                scene_results['status'] = 'failed'
                return scene_results

        # Stage 3: Evaluation
        if scene_config['evaluation']['enabled'] and strategy_results:
            try:
                eval_results = {}
                for strategy_name, stats in strategy_results.items():
                    eval_result = run_evaluation_stage(
                        config=scene_config,
                        inference_output_dir=stats['output_dir'],
                        strategy=strategy_name,
                    )
                    eval_results[strategy_name] = eval_result

                scene_results['evaluation'] = {
                    'status': 'success',
                    'results': eval_results
                }
            except Exception as e:
                logger.error(f"Evaluation failed for {scene_name}: {e}", exc_info=True)
                scene_results['evaluation'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        return scene_results

    except Exception as e:
        logger.error(f"Scene processing failed for {scene_name}: {e}", exc_info=True)
        scene_results['status'] = 'failed'
        scene_results['error'] = str(e)
        return scene_results


def generate_multi_scene_report(
    scene_results: List[Dict],
    aggregated: Dict,
    output_path: Path,
    config: Optional[Dict] = None
):
    """
    Generate multi-scene summary report with detailed statistics.

    Args:
        scene_results: List of scene result dictionaries
        aggregated: Aggregated metrics dictionary
        output_path: Path to save the report
        config: Configuration dictionary (for model info)
    """
    import numpy as np

    logger.info(f"Generating multi-scene summary report...")

    # Extract model info if available
    model_info = config.get('_model_info', {}) if config else {}
    model_type = model_info.get('type', 'unknown')
    model_name = model_info.get('model_name', 'unknown')
    pooling_mode = config.get('aggregation', {}).get('pooling_mode', 'unknown') if config else 'unknown'

    report_lines = [
        "# Multi-Scene Pipeline Summary Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Model Type**: {model_type}",
        f"- **Model Name**: `{model_name}`",
        f"- **Pooling Mode**: {pooling_mode}",
        f"- **Device**: {config.get('experiment', {}).get('device', 'unknown') if config else 'unknown'}",
        "",
        "## Overview",
        "",
        f"- **Total Scenes**: {aggregated['total_scenes']}",
        f"- **Successful**: {aggregated['successful_scenes']}",
        f"- **Failed**: {aggregated['failed_scenes']}",
        "",
    ]

    # Collect timing and proposal statistics
    timing_stats = {'aggregation': [], 'inference': [], 'total': []}
    proposal_stats = []

    for scene_result in scene_results:
        if scene_result['status'] != 'success':
            continue

        # Collect inference timing and proposals
        if 'inference' in scene_result and scene_result['inference'].get('status') == 'success':
            strategies = scene_result['inference'].get('strategies', {})
            for strategy_name, stats in strategies.items():
                if 'elapsed_time' in stats:
                    timing_stats['inference'].append(stats['elapsed_time'])
                if 'num_predictions' in stats:
                    proposal_stats.append(stats['num_predictions'])

    # Add timing statistics section
    report_lines.extend([
        "## Timing Statistics",
        "",
    ])

    if timing_stats['inference']:
        inf_times = timing_stats['inference']
        report_lines.extend([
            f"- **Inference Time (per scene)**:",
            f"  - Mean: {np.mean(inf_times):.2f}s",
            f"  - Std: {np.std(inf_times):.2f}s",
            f"  - Min: {np.min(inf_times):.2f}s",
            f"  - Max: {np.max(inf_times):.2f}s",
            f"  - Total: {np.sum(inf_times):.2f}s",
            "",
        ])
    else:
        report_lines.append("*No timing data available*\n")

    # Add proposal statistics section
    report_lines.extend([
        "## Proposal Statistics",
        "",
    ])

    if proposal_stats:
        report_lines.extend([
            f"- **Predictions per Scene**:",
            f"  - Mean: {np.mean(proposal_stats):.1f}",
            f"  - Std: {np.std(proposal_stats):.1f}",
            f"  - Min: {np.min(proposal_stats)}",
            f"  - Max: {np.max(proposal_stats)}",
            f"  - Total: {np.sum(proposal_stats)}",
            "",
        ])
    else:
        report_lines.append("*No proposal data available*\n")

    # Add overall metrics section
    report_lines.extend([
        "## Overall Metrics (Average across scenes)",
        "",
    ])

    if aggregated['overall_metrics']:
        report_lines.append("| Strategy | Mean mAP | Mean AP50 | Mean AP25 | Std mAP | Std AP50 | Std AP25 | Scenes |")
        report_lines.append("|----------|----------|-----------|-----------|---------|----------|----------|--------|")

        for strategy, metrics in aggregated['overall_metrics'].items():
            report_lines.append(
                f"| **{strategy}** | "
                f"{metrics['mean_mAP']:.4f} | "
                f"{metrics['mean_AP50']:.4f} | "
                f"{metrics['mean_AP25']:.4f} | "
                f"{metrics['std_mAP']:.4f} | "
                f"{metrics['std_AP50']:.4f} | "
                f"{metrics['std_AP25']:.4f} | "
                f"{metrics['num_scenes']} |"
            )
    else:
        report_lines.append("*No metrics available*")

    # Add detailed per-scene results table
    report_lines.extend([
        "",
        "## Per-Scene Detailed Results",
        "",
    ])

    # Create comprehensive table header
    report_lines.append("| Scene | Status | Predictions | Time (s) | mAP | AP50 | AP25 |")
    report_lines.append("|-------|--------|-------------|----------|-----|------|------|")

    for scene_result in scene_results:
        scene_name = scene_result['scene_name']
        status = scene_result['status']

        # Default values
        predictions_str = "-"
        time_str = "-"
        map_str = "-"
        ap50_str = "-"
        ap25_str = "-"

        if status == 'success':
            # Get predictions count and time from inference
            if 'inference' in scene_result and scene_result['inference'].get('status') == 'success':
                strategies = scene_result['inference'].get('strategies', {})
                # Use first strategy's data
                for strategy_name, stats in strategies.items():
                    predictions_str = str(stats.get('num_predictions', '-'))
                    if 'elapsed_time' in stats:
                        time_str = f"{stats['elapsed_time']:.1f}"
                    break

            # Get metrics from evaluation
            if 'evaluation' in scene_result and scene_result['evaluation'].get('status') == 'success':
                eval_results = scene_result['evaluation'].get('results', {})
                # Use first strategy's metrics
                for strategy_name, eval_result in eval_results.items():
                    if 'obj_part' in eval_result and eval_result['obj_part'].get('status') == 'success':
                        metrics = eval_result['obj_part']['metrics']
                        map_str = f"{metrics.get('all_ap', 0):.4f}"
                        ap50_str = f"{metrics.get('all_ap_50%', 0):.4f}"
                        ap25_str = f"{metrics.get('all_ap_25%', 0):.4f}"
                        break

        report_lines.append(
            f"| {scene_name} | {status} | {predictions_str} | {time_str} | {map_str} | {ap50_str} | {ap25_str} |"
        )

    # Add per-strategy breakdown if multiple strategies
    if aggregated['overall_metrics'] and len(aggregated['overall_metrics']) > 1:
        report_lines.extend([
            "",
            "## Per-Strategy Per-Scene Breakdown",
            "",
        ])

        for strategy in aggregated['overall_metrics'].keys():
            report_lines.extend([
                f"### {strategy.upper()} Strategy",
                "",
                "| Scene | mAP | AP50 | AP25 | Predictions | Time (s) |",
                "|-------|-----|------|------|-------------|----------|",
            ])

            for scene_result in scene_results:
                scene_name = scene_result['scene_name']
                if scene_result['status'] != 'success':
                    report_lines.append(f"| {scene_name} | - | - | - | - | - |")
                    continue

                # Get strategy-specific data
                predictions = "-"
                time_val = "-"
                map_val = "-"
                ap50_val = "-"
                ap25_val = "-"

                if 'inference' in scene_result and scene_result['inference'].get('status') == 'success':
                    strategies = scene_result['inference'].get('strategies', {})
                    if strategy in strategies:
                        stats = strategies[strategy]
                        predictions = str(stats.get('num_predictions', '-'))
                        if 'elapsed_time' in stats:
                            time_val = f"{stats['elapsed_time']:.1f}"

                if 'evaluation' in scene_result and scene_result['evaluation'].get('status') == 'success':
                    eval_results = scene_result['evaluation'].get('results', {})
                    if strategy in eval_results:
                        eval_result = eval_results[strategy]
                        if 'obj_part' in eval_result and eval_result['obj_part'].get('status') == 'success':
                            metrics = eval_result['obj_part']['metrics']
                            map_val = f"{metrics.get('all_ap', 0):.4f}"
                            ap50_val = f"{metrics.get('all_ap_50%', 0):.4f}"
                            ap25_val = f"{metrics.get('all_ap_25%', 0):.4f}"

                report_lines.append(
                    f"| {scene_name} | {map_val} | {ap50_val} | {ap25_val} | {predictions} | {time_val} |"
                )

            report_lines.append("")

    report_lines.extend([
        "---",
        "",
        f"Report generated by My3DIS multi-scene pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"  Multi-scene report saved: {output_path}")


def aggregate_scene_metrics(scene_results: List[Dict], output_path: Path):
    """
    Aggregate metrics from all scenes and save to file.

    Args:
        scene_results: List of scene result dictionaries
        output_path: Path to save aggregated metrics
    """
    import numpy as np  # Import numpy for metric aggregation

    logger.info("="*80)
    logger.info("AGGREGATING METRICS ACROSS ALL SCENES")
    logger.info("="*80)

    # Initialize aggregated metrics
    aggregated = {
        'total_scenes': len(scene_results),
        'successful_scenes': 0,
        'failed_scenes': 0,
        'scenes': {},
        'overall_metrics': {}
    }

    # Collect per-scene metrics
    all_eval_results = []

    for scene_result in scene_results:
        scene_name = scene_result['scene_name']
        aggregated['scenes'][scene_name] = {
            'status': scene_result['status'],
            'exp_dir': scene_result['exp_dir'],
            'scene_path': scene_result['scene_path']
        }

        if scene_result['status'] == 'success':
            aggregated['successful_scenes'] += 1

            # Extract evaluation metrics
            if 'evaluation' in scene_result and scene_result['evaluation'].get('status') == 'success':
                eval_results = scene_result['evaluation'].get('results', {})

                for strategy_name, eval_result in eval_results.items():
                    if 'obj_part' in eval_result and eval_result['obj_part'].get('status') == 'success':
                        metrics = eval_result['obj_part']['metrics']
                        all_eval_results.append({
                            'scene': scene_name,
                            'strategy': strategy_name,
                            'mAP': metrics.get('all_ap', 0),
                            'AP50': metrics.get('all_ap_50%', 0),
                            'AP25': metrics.get('all_ap_25%', 0),
                        })

                        # Store per-scene metrics
                        aggregated['scenes'][scene_name][strategy_name] = {
                            'mAP': metrics.get('all_ap', 0),
                            'AP50': metrics.get('all_ap_50%', 0),
                            'AP25': metrics.get('all_ap_25%', 0),
                        }
        else:
            aggregated['failed_scenes'] += 1
            if 'error' in scene_result:
                aggregated['scenes'][scene_name]['error'] = scene_result['error']

    # Compute overall metrics (average across scenes)
    if all_eval_results:
        # Group by strategy
        strategy_metrics = {}
        for result in all_eval_results:
            strategy = result['strategy']
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    'mAP': [],
                    'AP50': [],
                    'AP25': []
                }

            strategy_metrics[strategy]['mAP'].append(result['mAP'])
            strategy_metrics[strategy]['AP50'].append(result['AP50'])
            strategy_metrics[strategy]['AP25'].append(result['AP25'])

        # Compute averages (using nanmean/nanstd to handle potential NaN values)
        for strategy, metrics in strategy_metrics.items():
            # Convert to numpy arrays for proper NaN handling
            mAP_arr = np.array(metrics['mAP'])
            AP50_arr = np.array(metrics['AP50'])
            AP25_arr = np.array(metrics['AP25'])

            # Count valid (non-NaN) values
            valid_mAP = np.sum(~np.isnan(mAP_arr))
            valid_AP50 = np.sum(~np.isnan(AP50_arr))
            valid_AP25 = np.sum(~np.isnan(AP25_arr))

            # Use nanmean/nanstd to ignore NaN values
            aggregated['overall_metrics'][strategy] = {
                'mean_mAP': float(np.nanmean(mAP_arr)) if valid_mAP > 0 else 0.0,
                'mean_AP50': float(np.nanmean(AP50_arr)) if valid_AP50 > 0 else 0.0,
                'mean_AP25': float(np.nanmean(AP25_arr)) if valid_AP25 > 0 else 0.0,
                'std_mAP': float(np.nanstd(mAP_arr)) if valid_mAP > 0 else 0.0,
                'std_AP50': float(np.nanstd(AP50_arr)) if valid_AP50 > 0 else 0.0,
                'std_AP25': float(np.nanstd(AP25_arr)) if valid_AP25 > 0 else 0.0,
                'num_scenes': len(metrics['mAP']),
                'valid_scenes_mAP': int(valid_mAP),
                'valid_scenes_AP50': int(valid_AP50),
                'valid_scenes_AP25': int(valid_AP25),
            }

            # Log warning if some scenes had NaN values
            if valid_mAP < len(metrics['mAP']):
                logger.warning(f"  {strategy}: {len(metrics['mAP']) - valid_mAP} scenes had NaN mAP values")

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"✓ Aggregated metrics saved to {output_path}")
    logger.info(f"  Total scenes: {aggregated['total_scenes']}")
    logger.info(f"  Successful: {aggregated['successful_scenes']}")
    logger.info(f"  Failed: {aggregated['failed_scenes']}")

    # Print overall metrics
    for strategy, metrics in aggregated['overall_metrics'].items():
        logger.info(f"\n  {strategy.upper()} strategy (averaged over {metrics['num_scenes']} scenes):")
        logger.info(f"    mAP:  {metrics['mean_mAP']:.3f} ± {metrics['std_mAP']:.3f}")
        logger.info(f"    AP50: {metrics['mean_AP50']:.3f} ± {metrics['std_AP50']:.3f}")
        logger.info(f"    AP25: {metrics['mean_AP25']:.3f} ± {metrics['std_AP25']:.3f}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Run full My3DIS pipeline (Aggregation + Inference + Evaluation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config file (single scene mode)
  python scripts/run_full_pipeline.py --config configs/inference/full_pipeline.yaml

  # Run multi-scene pipeline
  python scripts/run_full_pipeline.py --config configs/inference/full_pipeline.yaml --multi-scene

  # Override experiment directory
  python scripts/run_full_pipeline.py \\
      --config configs/inference/full_pipeline.yaml \\
      --exp-dir /path/to/experiment

  # Skip aggregation if already done
  python scripts/run_full_pipeline.py \\
      --config configs/inference/full_pipeline.yaml \\
      --skip-aggregation
        """
    )

    parser.add_argument(
        '--config', required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--multi-scene', action='store_true',
        help='Enable multi-scene processing mode'
    )
    parser.add_argument(
        '--exp-dir',
        help='Override experiment directory path from config (single scene mode)'
    )
    parser.add_argument(
        '--exp-root',
        help='Override experiment root directory from config (multi-scene mode)'
    )
    parser.add_argument(
        '--scene-path',
        help='Override scene path from config (single scene mode)'
    )
    parser.add_argument(
        '--dataset-root',
        help='Override dataset root directory from config (multi-scene mode)'
    )
    parser.add_argument(
        '--scenes',
        nargs='*',
        help='Specific scenes to process in multi-scene mode (e.g., scene_00005_00 scene_00093_01)'
    )
    parser.add_argument(
        '--skip-aggregation', action='store_true',
        help='Skip aggregation stage (use existing aggregation output)'
    )
    parser.add_argument(
        '--skip-inference', action='store_true',
        help='Skip inference stage'
    )
    parser.add_argument(
        '--skip-evaluation', action='store_true',
        help='Skip evaluation stage'
    )
    parser.add_argument(
        '--gt-path',
        help='Override ground truth path from config (for evaluation)'
    )
    parser.add_argument(
        '--pooling-mode',
        choices=['average', 'voting'],
        help='Override feature pooling mode from config'
    )
    parser.add_argument(
        '--device',
        help='Override device (cuda, cuda:0, cuda:1, cpu)'
    )
    parser.add_argument(
        '--model',
        choices=['siglip', 'laion_clip', 'openai_clip'],
        help='Model type to use (overrides config model.type)'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Apply model preset (must be done before other overrides)
    config = apply_model_preset(config, args.model)

    # Apply command-line overrides
    if args.multi_scene:
        config['experiment']['multi_scene_mode'] = True
    if args.exp_dir:
        config['experiment']['exp_dir'] = args.exp_dir
    if args.exp_root:
        config['experiment']['exp_root'] = args.exp_root
    if args.scene_path:
        config['experiment']['scene_path'] = args.scene_path
    if args.dataset_root:
        config['experiment']['dataset_root'] = args.dataset_root
    if args.scenes:
        config['experiment']['scenes'] = args.scenes
    if args.skip_aggregation:
        config['aggregation']['enabled'] = False
    if args.skip_inference:
        config['inference']['enabled'] = False
    if args.skip_evaluation:
        config['evaluation']['enabled'] = False
    if args.gt_path:
        config['evaluation']['gt_path'] = args.gt_path
    if args.pooling_mode:
        config['aggregation']['pooling_mode'] = args.pooling_mode
    if args.device:
        config['experiment']['device'] = args.device

    # Check for multi-scene mode
    multi_scene_mode = config['experiment'].get('multi_scene_mode', False)

    if multi_scene_mode:
        # ===== MULTI-SCENE MODE =====
        logger.info("="*80)
        logger.info("MULTI-SCENE PROCESSING MODE")
        logger.info("="*80)

        exp_root = Path(config['experiment']['exp_root'])
        dataset_root = Path(config['experiment'].get('dataset_root', 'data/multiscan'))
        scenes_config = config['experiment'].get('scenes', 'all')

        logger.info(f"Experiment root: {exp_root}")
        logger.info(f"Dataset root: {dataset_root}")
        logger.info(f"Scenes config: {scenes_config}")

        # Discover scenes to process
        scene_list = discover_scenes(exp_root, scenes_config)

        if not scene_list:
            logger.error("No scenes found to process")
            return 1

        logger.info(f"\nProcessing {len(scene_list)} scenes:")
        for scene_name, scene_exp_dir, _ in scene_list:
            logger.info(f"  - {scene_name}")

        # Process each scene
        all_scene_results = []
        for i, (scene_name, scene_exp_dir, scene_data_path) in enumerate(scene_list, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"SCENE {i}/{len(scene_list)}: {scene_name}")
            logger.info(f"{'='*80}")

            scene_result = process_single_scene(
                scene_name=scene_name,
                scene_exp_dir=scene_exp_dir,
                scene_data_path=scene_data_path,
                config=config,
                dataset_root=dataset_root
            )
            all_scene_results.append(scene_result)

            # Save per-scene results
            scene_results_path = scene_exp_dir / "pipeline_results.json"
            with open(scene_results_path, 'w') as f:
                json.dump(scene_result, f, indent=2)
            logger.info(f"  Per-scene results saved: {scene_results_path}")

        # Aggregate metrics across all scenes
        # Use model-based naming to avoid overwriting different configurations
        model_info = config.get('_model_info', {})
        output_suffix = model_info.get('output_suffix', 'default')

        aggregated_metrics_path = exp_root / f"aggregated_metrics_{output_suffix}.json"
        aggregated = aggregate_scene_metrics(all_scene_results, aggregated_metrics_path)

        # Generate summary report with model-based naming
        summary_report_path = exp_root / f"multi_scene_summary_{output_suffix}.md"
        generate_multi_scene_report(all_scene_results, aggregated, summary_report_path, config)

        logger.info("")
        logger.info("="*80)
        logger.info("✓ MULTI-SCENE PIPELINE COMPLETED")
        logger.info("="*80)
        logger.info(f"Total scenes processed: {len(all_scene_results)}")
        logger.info(f"Successful: {aggregated['successful_scenes']}")
        logger.info(f"Failed: {aggregated['failed_scenes']}")
        logger.info(f"Aggregated metrics: {aggregated_metrics_path}")
        logger.info(f"Summary report: {summary_report_path}")

        return 0

    # ===== SINGLE SCENE MODE (Legacy) =====
    # Setup paths
    config = setup_paths(config)

    # Print configuration summary
    logger.info("="*80)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Experiment dir:   {config['experiment']['exp_dir']}")
    logger.info(f"Scene path:       {config['experiment']['scene_path']}")
    logger.info(f"Aggregation out:  {config['experiment']['aggregation_output_dir']}")
    logger.info(f"Inference out:    {config['experiment']['inference_output_dir']}")
    logger.info(f"Levels:           {config['experiment']['levels']}")
    logger.info(f"Device:           {config['experiment']['device']}")
    logger.info("")
    logger.info(f"Aggregation settings:")
    logger.info(f"  IoU threshold:   {config['aggregation']['iou_threshold']}")
    logger.info(f"  Area fraction:   {config['aggregation']['area_fraction']}")
    logger.info(f"  Pooling mode:    {config['aggregation']['pooling_mode']}")
    logger.info(f"  Extract features: {config['aggregation']['extract_features']}")
    logger.info(f"  SigLIP model:    {config['aggregation']['siglip_model']}")
    logger.info("")
    logger.info(f"Inference settings:")
    logger.info(f"  Multi-strategy testing: {config['inference'].get('test_all_strategies', False)}")
    if config['inference'].get('test_all_strategies', False):
        logger.info(f"  Strategies to test:     {config['inference'].get('strategies_to_test', [])}")
    else:
        logger.info(f"  Single strategy:        {config['inference']['strategy']}")
    logger.info(f"  NMS IoU:         {config['inference']['nms']['iou_threshold']}")
    logger.info(f"  Keep top-k:      {config['inference']['nms']['keep_top_k']}")
    logger.info("")
    logger.info("Evaluation settings:")
    logger.info(f"  GT path:         {config['evaluation']['gt_path']}")
    logger.info(f"  Query file:      {config['evaluation']['query_file']}")
    logger.info("")
    logger.info("Stages enabled:")
    logger.info(f"  Aggregation:    {config['aggregation']['enabled']}")
    logger.info(f"  Inference:      {config['inference']['enabled']}")
    logger.info(f"  Evaluation:     {config['evaluation']['enabled']}")
    logger.info("="*80)
    logger.info("")

    # Run pipeline stages
    proposal_data_paths = None
    strategy_results = {}

    try:
        # Stage 1: Aggregation
        if config['aggregation']['enabled']:
            proposal_data_paths = run_aggregation_stage(config)
        else:
            logger.info("Skipping aggregation stage (loading existing outputs)")
            # Load existing aggregation outputs
            agg_output_dir = Path(config['experiment']['aggregation_output_dir'])
            proposal_data_paths = {}

            # Auto-detect or use specified levels
            levels_to_check = config['experiment'].get('levels', None)
            if levels_to_check is None:
                from my3dis.aggregation.aggregation_pipeline import detect_available_levels
                exp_dir = Path(config['experiment']['exp_dir'])
                detected_levels = detect_available_levels(exp_dir)
                if detected_levels:
                    levels_to_check = detected_levels
                    logger.info(f"Auto-detected levels: {levels_to_check}")
                else:
                    levels_to_check = [2, 4, 6]
                    logger.warning(f"No levels detected, using default: {levels_to_check}")

            for level in levels_to_check:
                npz_path = agg_output_dir / f"proposal_data_level{level}.npz"
                if npz_path.exists():
                    proposal_data_paths[str(level)] = str(npz_path)
                else:
                    logger.warning(f"Aggregation output not found: {npz_path}")

        # Stage 2: Inference
        if config['inference']['enabled']:
            if not proposal_data_paths:
                logger.error("No aggregation outputs available for inference")
                return 1

            # Check if multi-strategy testing is enabled
            test_all_strategies = config['inference'].get('test_all_strategies', False)

            if test_all_strategies:
                # Multi-strategy testing mode
                strategies_to_test = config['inference'].get('strategies_to_test', ['independent', 'hierarchical'])
                logger.info("")
                logger.info("="*80)
                logger.info(f"MULTI-STRATEGY TESTING MODE: Testing {len(strategies_to_test)} strategies")
                logger.info("="*80)

                for strategy in strategies_to_test:
                    logger.info("")
                    logger.info(f"--- Testing strategy: {strategy} ---")

                    # Run inference with this strategy
                    inference_stats = run_inference_stage(
                        config,
                        proposal_data_paths,
                        strategy=strategy,
                        output_subdir=strategy
                    )
                    strategy_results[strategy] = inference_stats

                # Generate comparison report
                logger.info("")
                logger.info("="*80)
                logger.info("GENERATING STRATEGY COMPARISON REPORT")
                logger.info("="*80)

                comparison_report_path = Path(config['experiment']['inference_output_dir']) / "strategy_comparison.md"
                generate_comparison_report(
                    strategy_results=strategy_results,
                    config=config,
                    output_path=str(comparison_report_path)
                )

            else:
                # Single strategy mode
                logger.info("")
                logger.info("Single strategy mode")
                inference_stats = run_inference_stage(config, proposal_data_paths)
                strategy_name = config['inference']['strategy']
                strategy_results[strategy_name] = inference_stats

        # Stage 3: Evaluation
        if config['evaluation']['enabled']:
            if not strategy_results:
                logger.error("No predictions available for evaluation")
                return 1

            # Infer num_points from first NPZ file
            first_level = config['experiment']['levels'][0]
            agg_output_dir = Path(config['experiment']['aggregation_output_dir'])
            first_npz = agg_output_dir / f"proposal_data_level{first_level}.npz"
            if first_npz.exists():
                import numpy as np
                data = np.load(first_npz)
                num_points = data['proposal_masks'].shape[0]
                logger.info(f"Detected num_points={num_points} from aggregation output")
            else:
                num_points = None
                logger.warning("Could not infer num_points, evaluation may fail")

            # Run evaluation for each strategy
            eval_results = {}
            for strategy_name, stats in strategy_results.items():
                logger.info(f"\nEvaluating strategy: {strategy_name}")
                eval_result = run_evaluation_stage(
                    config=config,
                    inference_output_dir=stats['output_dir'],
                    strategy=strategy_name,
                )
                eval_results[strategy_name] = eval_result

                # Log evaluation results (updated for multi-mode)
                if eval_result.get('summary', {}).get('successful', 0) > 0:
                    logger.info(f"  ✓ {strategy_name} evaluation completed:")
                    for mode, result in eval_result.items():
                        if mode != 'summary' and result.get('status') == 'success':
                            logger.info(f"    {mode}: mAP={result['mAP']:.3f}, AP50={result['AP50']:.3f}, AP25={result['AP25']:.3f}")
                else:
                    logger.warning(f"  ✗ {strategy_name}: No successful evaluations")

                # Generate strategy report with evaluation results
                strategy_report_path = Path(stats['output_dir']) / f"{strategy_name}_report.md"
                generate_strategy_report(
                    strategy_name=strategy_name,
                    inference_stats=stats,
                    config=config,
                    output_path=str(strategy_report_path),
                    eval_results=eval_result
                )

        logger.info("")
        logger.info("="*80)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
