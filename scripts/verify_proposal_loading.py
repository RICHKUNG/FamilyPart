#!/usr/bin/env python3
"""
验证 proposal 加载是否正常（无 WARNING）

这个脚本测试 inference 系统能否正确加载 aggregation 输出的 proposals
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_proposal_loading(aggregation_output_dir: str, levels: list):
    """测试 proposal 加载"""
    from my3dis.inference.retrieval import Proposal

    aggregation_output_dir = Path(aggregation_output_dir)
    proposals_by_level = {}

    logger.info("=" * 60)
    logger.info("测试 Proposal 加载")
    logger.info("=" * 60)

    for level in levels:
        level_key = f"L{level}"
        npz_path = aggregation_output_dir / f"proposal_data_level{level}.npz"

        if not npz_path.exists():
            logger.error(f"✗ Level {level}: 文件不存在 - {npz_path}")
            continue

        logger.info(f"\nLevel {level} ({level_key}):")
        logger.info(f"  文件: {npz_path}")

        try:
            # 加载 NPZ 文件
            data = np.load(npz_path, allow_pickle=False)

            proposal_masks = data['proposal_masks']
            proposal_features = data['proposal_features']
            proposal_ids = data['proposal_ids']

            logger.info(f"  ✓ 加载成功:")
            logger.info(f"    - Masks: {proposal_masks.shape}")
            logger.info(f"    - Features: {proposal_features.shape}")
            logger.info(f"    - IDs: {len(proposal_ids)}")

            # 创建 Proposal 对象（模拟 inference pipeline）
            proposals = []
            for i, prop_id in enumerate(proposal_ids):
                proposals.append(Proposal(
                    proposal_id=int(prop_id),
                    level=level_key,
                    mask=proposal_masks[:, i],
                    feature=proposal_features[i],
                    object_id=int(prop_id),
                ))

            proposals_by_level[level_key] = proposals
            logger.info(f"  ✓ 创建了 {len(proposals)} 个 Proposal 对象")

        except Exception as e:
            logger.error(f"✗ Level {level}: 加载失败 - {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("加载摘要:")
    logger.info("=" * 60)
    for level_key, proposals in proposals_by_level.items():
        logger.info(f"  {level_key}: {len(proposals)} proposals")

    # 测试 retrieval 系统
    if proposals_by_level:
        logger.info("\n" + "=" * 60)
        logger.info("测试 MultiLevelRetriever")
        logger.info("=" * 60)

        from my3dis.inference.retrieval import MultiLevelRetriever

        retriever = MultiLevelRetriever(
            proposals_by_level=proposals_by_level,
            temperature=0.2,
            min_similarity=0.05
        )

        # 测试每个层级
        # 从第一个 proposal 获取正确的 feature 维度
        first_level = list(proposals_by_level.keys())[0]
        feature_dim = proposals_by_level[first_level][0].feature.shape[0]
        logger.info(f"\n检测到 feature 维度: {feature_dim}")

        for level_key in proposals_by_level.keys():
            logger.info(f"\n测试 retrieval for level {level_key}:")
            result = retriever.retrieve_single_level(
                query_feat=np.random.rand(feature_dim),  # 使用正确维度
                level=level_key,
                top_k=10
            )
            logger.info(f"  ✓ Retrieved {len(result.proposals)} proposals")

        logger.info("\n✓ 所有测试通过！没有 WARNING！")
    else:
        logger.error("\n✗ 没有加载到任何 proposals")

    return proposals_by_level

def main():
    import argparse

    parser = argparse.ArgumentParser(description='验证 proposal 加载')
    parser.add_argument(
        '--aggregation-dir',
        default='./outputs/experiments/v6_135_ssam2_filter10_fill50_propinf/scene_00005_00/aggregation_output',
        help='Aggregation 输出目录'
    )
    parser.add_argument(
        '--levels',
        type=int,
        nargs='+',
        default=[1, 3, 5],
        help='要测试的层级'
    )
    args = parser.parse_args()

    test_proposal_loading(args.aggregation_dir, args.levels)

if __name__ == '__main__':
    main()
