#!/usr/bin/env python3
"""
Quick evaluation script for My3DIS experiments on MultiScan dataset.

This script performs the complete pipeline:
1. Aggregation: Load multi-level tracking outputs (L2, L4, L6) using ProposalLoader
2. Feature Extraction: Compute SigLIP features for each proposal using SigLIPFeatureExtractor
3. Inference: Run OV-3DIS inference with multi-instance handling
4. Evaluation: Compute mAP using Search3D evaluation protocol

Usage:
    python scripts/quick_eval_experiment.py \
        --exp-dir ./outputs/experiments/test_1105_ssam4_filter300_fill500_fix \
        --dataset-root data/multiscan \
        --gt-path /path/to/multiscan_annotations_search3d/ov_part_annotations \
        --output-dir /tmp/eval_results \
        --siglip-model google/siglip-so400m-patch14-384 \
        --strategy independent \
        --num-workers 4

Features:
- Multi-level proposal aggregation from My3DIS tracking outputs using ProposalLoader
- Efficient SigLIP feature extraction with batching using SigLIPFeatureExtractor
- Multi-instance aware inference using InferencePipeline
- Full Search3D evaluation with mAP@25, mAP@50, mAP
- Detailed per-scene and per-class metrics

Refactored Version:
- Uses my3dis.aggregation.ProposalLoader instead of custom ExperimentAggregator
- Uses my3dis.inference.SigLIPFeatureExtractor instead of inline implementation
- Cleaner separation of concerns and better code reuse
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3DprojToSiglip"))

import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# My3DIS imports - using refactored modules
from my3dis.aggregation import ProposalLoader
from my3dis.inference import (
    InferencePipeline,
    SigLIPFeatureExtractor,
)
from my3dis.inference.config import InferenceConfig
from my3dis.inference.retrieval import Proposal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickEvaluator:
    """
    End-to-end evaluator for My3DIS experiments.

    This class orchestrates the full evaluation pipeline using My3DIS modules:
    - ProposalLoader: Load proposals from tracking outputs
    - SigLIPFeatureExtractor: Extract features
    - InferencePipeline: Run inference with multi-instance handling
    - Search3D evaluation: Compute mAP metrics
    """

    def __init__(
        self,
        config: InferenceConfig,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda"
    ):
        """
        Initialize evaluator.

        Args:
            config: Inference configuration
            siglip_model: SigLIP model name
            device: Device to run on
        """
        self.config = config
        self.device = device

        # Initialize feature extractor
        self.feature_extractor = SigLIPFeatureExtractor(siglip_model, device)

    def load_search3d_queries(
        self,
        query_file: Optional[str] = None
    ) -> List[Tuple[str, str, int]]:
        """
        Load Search3D OV-Part queries.

        Returns:
            List of (object_text, part_text, label_id) tuples
        """
        # Default queries from Search3D paper (object-part pairs)
        default_queries = [
            ("cabinet", "door", 121003),
            ("cabinet", "drawer", 121009),
            ("table", "leg", 134018),
            ("chair", "leg", 104018),
            ("chair", "back", 104002),
            ("bed", "frame", 107013),
            ("sofa", "cushion", 132008),
            ("shelf", "board", 129004),
        ]

        if query_file is not None and Path(query_file).exists():
            with open(query_file, 'r') as f:
                data = json.load(f)
            queries = [(q['object'], q['part'], q['label_id']) for q in data]
            logger.info(f"Loaded {len(queries)} queries from {query_file}")
        else:
            queries = default_queries
            logger.info(f"Using {len(queries)} default queries")

        return queries

    def evaluate_experiment(
        self,
        exp_dir: str,
        dataset_root: str,
        gt_path: str,
        queries: Optional[List[Tuple[str, str, int]]] = None,
        levels: List[str] = ['L2', 'L4', 'L6']
    ) -> Dict:
        """
        Run full evaluation pipeline for an experiment.

        Args:
            exp_dir: Experiment directory path
            dataset_root: MultiScan dataset root
            gt_path: Ground truth annotations path
            queries: Optional list of queries (uses default if None)
            levels: Levels to use

        Returns:
            Evaluation results dict
        """
        logger.info(f"Evaluating experiment: {exp_dir}")

        # 1. Load proposals using ProposalLoader
        logger.info("Step 1/4: Loading proposals from tracking outputs")
        proposal_loader = ProposalLoader(exp_dir, dataset_root)
        proposals_raw, num_points, scene_name = proposal_loader.load_proposals(
            levels=levels,
            aggregation_method='union'
        )

        if not proposals_raw:
            raise ValueError("No proposals loaded from tracking outputs")

        logger.info(f"Loaded proposals: {[(level, len(props)) for level, props in proposals_raw.items()]}")

        # 2. Extract features using SigLIPFeatureExtractor
        logger.info("Step 2/4: Extracting SigLIP features")
        scene_path = Path(dataset_root) / scene_name
        features_by_level = self.feature_extractor.extract_proposal_features(
            proposals_raw, Path(exp_dir), scene_path
        )

        # 3. Prepare proposals for inference
        logger.info("Step 3/4: Preparing proposals for inference")
        proposals_by_level = {}
        for level in proposals_raw.keys():
            proposals = []
            for i, (prop_data, feature) in enumerate(zip(proposals_raw[level], features_by_level[level])):
                proposal = Proposal(
                    level=level,
                    proposal_id=i,
                    mask=prop_data['mask'],
                    feature=feature.cpu().numpy(),  # Move to CPU before converting to numpy
                    object_id=prop_data['object_id']
                )
                proposals.append(proposal)
            proposals_by_level[level] = proposals

        logger.info(f"Prepared {sum(len(props) for props in proposals_by_level.values())} proposals")

        # Load family tree if available
        family_tree_path = Path(exp_dir) / "relations" / "family_tree.json"
        if not family_tree_path.exists():
            logger.warning(f"Family tree not found: {family_tree_path}")
            family_tree_path = None
        else:
            logger.info(f"Using family tree: {family_tree_path}")

        # Initialize inference pipeline
        pipeline = InferencePipeline(
            proposals_by_level=proposals_by_level,
            family_tree_path=str(family_tree_path) if family_tree_path else None,
            strategy=self.config.strategy,
            num_points=num_points,
            **self.config.get_strategy_kwargs()
        )

        # Load queries
        if queries is None:
            queries = self.load_search3d_queries()

        # Get query embeddings
        logger.info(f"Processing {len(queries)} queries")
        object_texts = [q[0] for q in queries]
        part_texts = [q[1] for q in queries]
        label_ids = [q[2] for q in queries]

        object_features = self.feature_extractor.extract_text_features(object_texts)
        part_features = self.feature_extractor.extract_text_features(part_texts)

        # Run inference
        all_predictions = []
        for i, label_id in enumerate(tqdm(label_ids, desc="Running inference")):
            predictions = pipeline.infer_single_query(
                object_query_feat=object_features[i].numpy(),
                part_query_feat=part_features[i].numpy(),
                label_id=label_id,
                nms_iou_threshold=self.config.nms.iou_threshold,
                use_soft_nms=self.config.nms.use_soft_nms,
                keep_top_k=self.config.nms.keep_top_k
            )
            all_predictions.extend(predictions)

        logger.info(f"Generated {len(all_predictions)} predictions")

        # Format predictions
        formatted = pipeline.format_predictions(all_predictions)
        predictions_dict = {scene_name: formatted}

        # 4. Evaluate with Search3D
        logger.info("Step 4/4: Running Search3D evaluation")
        results = self.run_search3d_evaluation(predictions_dict, gt_path)

        return {
            'scene_name': scene_name,
            'num_proposals': {level: len(props) for level, props in proposals_by_level.items()},
            'num_predictions': len(all_predictions),
            'metrics': results
        }

    def run_search3d_evaluation(
        self,
        predictions: Dict,
        gt_path: str
    ) -> Dict:
        """
        Run Search3D evaluation protocol using local evaluation code.

        Args:
            predictions: Dict mapping scene names to prediction dicts
            gt_path: Path to ground truth annotations

        Returns:
            Evaluation metrics dict
        """
        try:
            # Import local Search3D evaluation code
            eval_dir = Path(__file__).parent.parent / "eval" / "search3d"
            if not eval_dir.exists():
                raise ImportError(f"Search3D evaluation code not found at {eval_dir}")

            sys.path.insert(0, str(eval_dir))
            from eval_semantic_instance_parts_OV import evaluate

            output_file = "/tmp/search3d_results.csv"
            avgs = evaluate(
                predictions,
                gt_path=gt_path,
                output_file=output_file,
                dataset="multiscan_search3d_ov_parts"
            )

            return {
                'mAP': avgs.get('all_ap', 0.0),
                'mAP@50': avgs.get('all_ap_50%', 0.0),
                'mAP@25': avgs.get('all_ap_25%', 0.0),
                'details': avgs
            }

        except ImportError as e:
            logger.error(f"Search3D evaluation not available: {e}")
            logger.info("Evaluation code should be in eval/search3d/")
            logger.info("Download GT data: see scripts/DOWNLOAD_GT_INSTRUCTIONS.md")

            # Return dummy metrics
            return {
                'mAP': 0.0,
                'mAP@50': 0.0,
                'mAP@25': 0.0,
                'error': f'Search3D evaluation not available: {e}'
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
                'mAP': 0.0,
                'mAP@50': 0.0,
                'mAP@25': 0.0,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation for My3DIS experiments on MultiScan"
    )

    # Required arguments
    parser.add_argument(
        '--exp-dir', required=True,
        help='Path to experiment directory (e.g., outputs/experiments/scene_XX/run_YY)'
    )
    parser.add_argument(
        '--dataset-root', required=True,
        help='MultiScan dataset root directory'
    )
    parser.add_argument(
        '--gt-path', required=True,
        help='Path to Search3D ground truth annotations'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir', default='/tmp/eval_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to inference config YAML (optional)'
    )
    parser.add_argument(
        '--siglip-model', default='google/siglip-so400m-patch14-384',
        help='SigLIP model name'
    )
    parser.add_argument(
        '--strategy', default='independent', choices=['independent', 'hierarchical'],
        help='Inference strategy'
    )
    parser.add_argument(
        '--levels', nargs='+', default=['L2', 'L4', 'L6'],
        help='Levels to use'
    )
    parser.add_argument(
        '--queries', default=None,
        help='Path to custom queries JSON file'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if args.config and Path(args.config).exists():
        config = InferenceConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        # Use default config
        config = InferenceConfig(strategy=args.strategy)
        logger.info("Using default config")

    # Initialize evaluator
    evaluator = QuickEvaluator(
        config=config,
        siglip_model=args.siglip_model,
        device=args.device
    )

    # Load custom queries if provided
    queries = None
    if args.queries:
        queries = evaluator.load_search3d_queries(args.queries)

    # Run evaluation
    try:
        results = evaluator.evaluate_experiment(
            exp_dir=args.exp_dir,
            dataset_root=args.dataset_root,
            gt_path=args.gt_path,
            queries=queries,
            levels=args.levels
        )

        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Scene: {results['scene_name']}")
        print(f"Proposals: {results['num_proposals']}")
        print(f"Predictions: {results['num_predictions']}")
        print("\nMetrics:")
        print(f"  mAP:    {results['metrics']['mAP']:.4f}")
        print(f"  mAP@50: {results['metrics']['mAP@50']:.4f}")
        print(f"  mAP@25: {results['metrics']['mAP@25']:.4f}")
        print("="*60 + "\n")

        # Save results
        results_file = output_dir / f"{results['scene_name']}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
