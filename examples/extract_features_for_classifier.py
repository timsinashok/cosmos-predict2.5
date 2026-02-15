#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract features from images/videos for classifier training.

This script demonstrates how to extract latent representations from a dataset
and save them for training a downstream classifier.

Example usage:
    # Extract features from a directory of images
    python extract_features_for_classifier.py \\
        --experiment your-experiment \\
        --ckpt_path path/to/checkpoint \\
        --input_dir path/to/images \\
        --output_file features.pt
    
    # Then train a classifier
    python train_classifier.py --features features.pt
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from cosmos_predict2._src.predict2.inference.image2representation import Image2RepresentationExtractor


def extract_features_from_directory(
    extractor: Image2RepresentationExtractor,
    input_dir: Path,
    output_file: Path,
    batch_size: int = 4,
    resolution: tuple[int, int] = (360, 640),
    pool_method: str = "mean",
):
    """
    Extract features from all images/videos in a directory.
    
    Args:
        extractor: Initialized feature extractor
        input_dir: Directory containing input files
        output_file: Where to save extracted features
        batch_size: Batch size for processing
        resolution: Target resolution
        pool_method: Pooling method for features
    """
    # Find all image/video files
    image_exts = ['.jpg', '.jpeg', '.png', '.webp']
    video_exts = ['.mp4', '.avi', '.mov']
    
    all_files = []
    for ext in image_exts + video_exts:
        all_files.extend(list(input_dir.rglob(f'*{ext}')))
    
    print(f"Found {len(all_files)} files in {input_dir}")
    
    if not all_files:
        print("No files found!")
        return
    
    # Extract features in batches
    all_features = []
    all_paths = []
    
    pbar = tqdm(total=len(all_files), desc="Extracting features")
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        
        try:
            # Extract features for batch
            batch_features = []
            for file_path in batch_files:
                feature = extractor.extract_pooled(
                    file_path,
                    resolution=resolution,
                    pool_method=pool_method,
                )
                batch_features.append(feature)
            
            # Store results
            batch_features = torch.cat(batch_features, dim=0)
            all_features.append(batch_features)
            all_paths.extend([str(f) for f in batch_files])
            
            pbar.update(len(batch_files))
            
        except Exception as e:
            print(f"\nError processing batch starting at {batch_files[0]}: {e}")
            pbar.update(len(batch_files))
            continue
    
    pbar.close()
    
    # Concatenate all features
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        
        # Save to file
        output_data = {
            'features': all_features,
            'file_paths': all_paths,
            'resolution': resolution,
            'pool_method': pool_method,
        }
        
        torch.save(output_data, output_file)
        
        print(f"\n✓ Saved {len(all_paths)} features to {output_file}")
        print(f"  Features shape: {tuple(all_features.shape)}")
        print(f"  File size: {output_file.stat().st_size / (1024**2):.2f} MB")
    else:
        print("\n❌ No features extracted!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for classifier training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Input/output
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images/videos')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file for features (.pt)')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--resolution', type=str, default='360,640',
                       help='Target resolution (H,W)')
    parser.add_argument('--pool_method', type=str, default='mean',
                       choices=['mean', 'max', 'flatten'],
                       help='Pooling method for features')
    
    args = parser.parse_args()
    
    # Parse resolution
    target_h, target_w = map(int, args.resolution.split(','))
    resolution = (target_h, target_w)
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FEATURE EXTRACTION FOR CLASSIFIER TRAINING")
    print("="*80)
    print(f"\nExperiment:   {args.experiment}")
    print(f"Checkpoint:   {args.ckpt_path}")
    print(f"Input dir:    {input_dir}")
    print(f"Output file:  {output_file}")
    print(f"Resolution:   {target_h}x{target_w}")
    print(f"Pool method:  {args.pool_method}")
    print(f"Batch size:   {args.batch_size}")
    
    # Initialize extractor
    print(f"\n{'='*80}")
    print("Initializing extractor...")
    print(f"{'='*80}")
    
    extractor = Image2RepresentationExtractor(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
    )
    
    print("\n✓ Extractor initialized")
    
    # Extract features
    print(f"\n{'='*80}")
    print("Extracting features...")
    print(f"{'='*80}\n")
    
    extract_features_from_directory(
        extractor=extractor,
        input_dir=input_dir,
        output_file=output_file,
        batch_size=args.batch_size,
        resolution=resolution,
        pool_method=args.pool_method,
    )
    
    print(f"\n{'='*80}")
    print("✓ EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print("\nNext steps:")
    print(f"  1. Load features: data = torch.load('{output_file}')")
    print(f"  2. Access features: features = data['features']")
    print(f"  3. Access paths: paths = data['file_paths']")
    print(f"  4. Train classifier on features")


if __name__ == "__main__":
    main()
