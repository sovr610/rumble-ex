#!/usr/bin/env python3
"""
Full Training Pipeline for Brain-Inspired AI

Orchestrates training across all 7 phases:
1. SNN Core
2. Encoders
3. HTM
4. Global Workspace
5. Active Inference
6. Reasoning
7. Meta-Learning

Usage:
    # Development (quick validation)
    python scripts/train_full_pipeline.py --mode dev
    
    # Production 7B
    python scripts/train_full_pipeline.py --mode production --use-amp
    
    # Resume from specific phase
    python scripts/train_full_pipeline.py --mode production --start-phase 4
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PHASES = [
    ('Phase 1: SNN Core', 'train_phase1.py'),
    ('Phase 2: Encoders', 'train_phase2.py'),
    ('Phase 3: HTM', 'train_phase3.py'),
    ('Phase 4: Global Workspace', 'train_phase4.py'),
    ('Phase 5: Active Inference', 'train_phase5.py'),
    ('Phase 6: Reasoning', 'train_phase6.py'),
    ('Phase 7: Meta-Learning', 'train_phase7.py'),
]


def run_phase(
    phase_name: str,
    script_name: str,
    mode: str,
    extra_args: list,
    scripts_dir: Path,
):
    """Run a single training phase."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {phase_name}")
    logger.info(f"{'='*60}")
    
    script_path = scripts_dir / script_name
    
    if not script_path.exists():
        logger.warning(f"Script {script_path} not found, skipping...")
        return False
    
    cmd = [
        sys.executable, str(script_path),
        '--mode', mode,
    ] + extra_args
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(scripts_dir.parent),
        )
        logger.info(f"✅ {phase_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {phase_name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Full Training Pipeline')
    parser.add_argument('--mode', type=str, default='dev',
                        choices=['dev', 'production', 'production_3b',
                                 'production_1b'])
    parser.add_argument('--start-phase', type=int, default=1,
                        help='Phase to start from (1-7)')
    parser.add_argument('--end-phase', type=int, default=7,
                        help='Phase to end at (1-7)')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--checkpoint-base', type=str, default='checkpoints',
                        help='Base checkpoint directory')
    
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).parent
    
    # Build extra arguments
    extra_args = []
    if args.use_amp:
        extra_args.append('--use-amp')
    if args.compile:
        extra_args.append('--compile')
    
    # Training summary
    logger.info("\n" + "🧠" * 30)
    logger.info("  BRAIN-INSPIRED AI - FULL TRAINING PIPELINE")
    logger.info("🧠" * 30)
    logger.info(f"\nMode: {args.mode}")
    logger.info(f"Phases: {args.start_phase} to {args.end_phase}")
    logger.info(f"AMP: {args.use_amp}")
    logger.info(f"Compile: {args.compile}")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Run phases
    results = []
    for i, (phase_name, script_name) in enumerate(PHASES):
        phase_num = i + 1
        
        if phase_num < args.start_phase:
            logger.info(f"Skipping {phase_name} (before start phase)")
            continue
        
        if phase_num > args.end_phase:
            logger.info(f"Skipping {phase_name} (after end phase)")
            continue
        
        success = run_phase(
            phase_name, script_name, args.mode, extra_args, scripts_dir
        )
        results.append((phase_name, success))
        
        if not success and args.mode == 'production':
            logger.error("Production training failed, stopping pipeline")
            break
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    for phase_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {phase_name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} phases completed")
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
