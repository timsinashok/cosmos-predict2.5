#!/usr/bin/env python3
"""
Test script for Image2Representation extraction.

This script tests the new image-to-representation pipeline that bypasses
the diffusion model and extracts features directly for classifier training.

Usage:
    python test_embedding_generation.py \
        --experiment your-experiment \
        --ckpt_path path/to/checkpoint \
        --image path/to/image.jpg \
        --benchmark
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from cosmos_predict2._src.predict2.inference.image2representation import Image2RepresentationExtractor


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print detailed information about a tensor."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Shape:      {tuple(tensor.shape)}")
    print(f"Dtype:      {tensor.dtype}")
    print(f"Device:     {tensor.device}")
    print(f"Size (MB):  {tensor.element_size() * tensor.nelement() / (1024**2):.2f}")
    print(f"Min/Max:    {tensor.min().item():.4f} / {tensor.max().item():.4f}")
    print(f"Mean/Std:   {tensor.mean().item():.4f} / {tensor.std().item():.4f}")
    
    # Count non-zero elements
    nonzero = (tensor != 0).sum().item()
    total = tensor.nelement()
    print(f"Non-zero:   {nonzero:,} / {total:,} ({nonzero/total*100:.1f}%)")


def test_single_image(extractor: Image2RepresentationExtractor, image_path: str):
    """Test extraction from a single image."""
    print(f"\n{'='*80}")
    print("TEST: Single Image Extraction")
    print(f"{'='*80}")
    print(f"Image: {image_path}")
    
    # Test basic extraction
    print("\n[1/3] Extracting latent representation...")
    start = time.time()
    latent = extractor.extract_from_image(image_path)
    elapsed = time.time() - start
    
    print(f"✓ Extracted in {format_time(elapsed)}")
    print_tensor_info("Latent Representation", latent)
    
    # Test pooled extraction (mean)
    print("\n[2/3] Extracting pooled feature vector (mean)...")
    start = time.time()
    pooled_mean = extractor.extract_pooled(image_path, pool_method="mean")
    elapsed = time.time() - start
    
    print(f"✓ Extracted in {format_time(elapsed)}")
    print_tensor_info("Pooled Features (Mean)", pooled_mean)
    
    # Test pooled extraction (max)
    print("\n[3/3] Extracting pooled feature vector (max)...")
    start = time.time()
    pooled_max = extractor.extract_pooled(image_path, pool_method="max")
    elapsed = time.time() - start
    
    print(f"✓ Extracted in {format_time(elapsed)}")
    print_tensor_info("Pooled Features (Max)", pooled_max)
    
    return {
        'latent': latent,
        'pooled_mean': pooled_mean,
        'pooled_max': pooled_max,
    }


def test_single_video(extractor: Image2RepresentationExtractor, video_path: str):
    """Test extraction from a single video."""
    print(f"\n{'='*80}")
    print("TEST: Single Video Extraction")
    print(f"{'='*80}")
    print(f"Video: {video_path}")
    
    print("\n[1/2] Extracting latent representation...")
    start = time.time()
    latent = extractor.extract_from_video(video_path)
    elapsed = time.time() - start
    
    print(f"✓ Extracted in {format_time(elapsed)}")
    print_tensor_info("Video Latent Representation", latent)
    
    print("\n[2/2] Extracting pooled feature vector...")
    start = time.time()
    pooled = extractor.extract_pooled(video_path, pool_method="mean")
    elapsed = time.time() - start
    
    print(f"✓ Extracted in {format_time(elapsed)}")
    print_tensor_info("Pooled Video Features", pooled)
    
    return {
        'latent': latent,
        'pooled': pooled,
    }


def benchmark_speed(extractor: Image2RepresentationExtractor, test_image: str, num_iterations: int = 10):
    """Benchmark extraction speed."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: Speed Test ({num_iterations} iterations)")
    print(f"{'='*80}")
    
    # Warmup
    print("\nWarming up GPU...")
    for _ in range(3):
        _ = extractor.extract_from_image(test_image)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark full latent extraction
    print(f"\n[1/2] Benchmarking full latent extraction...")
    times_latent = []
    for i in range(num_iterations):
        start = time.time()
        latent = extractor.extract_from_image(test_image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times_latent.append(elapsed)
        print(f"  Iteration {i+1}/{num_iterations}: {format_time(elapsed)}")
    
    mean_latent = np.mean(times_latent)
    std_latent = np.std(times_latent)
    print(f"\n  Mean: {format_time(mean_latent)} ± {format_time(std_latent)}")
    print(f"  Throughput: {1/mean_latent:.2f} images/sec")
    
    # Benchmark pooled extraction
    print(f"\n[2/2] Benchmarking pooled feature extraction...")
    times_pooled = []
    for i in range(num_iterations):
        start = time.time()
        pooled = extractor.extract_pooled(test_image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times_pooled.append(elapsed)
        print(f"  Iteration {i+1}/{num_iterations}: {format_time(elapsed)}")
    
    mean_pooled = np.mean(times_pooled)
    std_pooled = np.std(times_pooled)
    print(f"\n  Mean: {format_time(mean_pooled)} ± {format_time(std_pooled)}")
    print(f"  Throughput: {1/mean_pooled:.2f} images/sec")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Full Latent:     {format_time(mean_latent)} ± {format_time(std_latent)}")
    print(f"Pooled Features: {format_time(mean_pooled)} ± {format_time(std_pooled)}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        print(f"  Peak:      {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")


def test_batch_processing(extractor: Image2RepresentationExtractor, test_image: str, batch_sizes: list[int] = [1, 2, 4, 8]):
    """Test batch processing with different batch sizes."""
    print(f"\n{'='*80}")
    print("TEST: Batch Processing")
    print(f"{'='*80}")
    
    # Create a list of the same image (for testing purposes)
    for batch_size in batch_sizes:
        print(f"\n[Batch size: {batch_size}]")
        input_paths = [test_image] * batch_size
        
        start = time.time()
        latents = extractor.extract_batch(input_paths, batch_size=batch_size)
        elapsed = time.time() - start
        
        per_image = elapsed / batch_size
        throughput = batch_size / elapsed
        
        print(f"  Total time:     {format_time(elapsed)}")
        print(f"  Per image:      {format_time(per_image)}")
        print(f"  Throughput:     {throughput:.2f} images/sec")
        print(f"  Output shape:   {tuple(latents.shape)}")


def test_classifier_pipeline(extractor: Image2RepresentationExtractor, test_image: str):
    """Demonstrate how to use this for classifier training."""
    print(f"\n{'='*80}")
    print("DEMO: Classifier Training Pipeline")
    print(f"{'='*80}")
    
    # Extract features for classifier
    print("\n[Step 1] Extract features from image")
    features = extractor.extract_pooled(test_image, pool_method="mean")
    print(f"  Feature vector shape: {tuple(features.shape)}")
    print(f"  Feature vector size:  {features.shape[1]} dimensions")
    
    # Simulate a simple classifier
    print("\n[Step 2] Example: Train a classifier on these features")
    print(f"  >>> features = extractor.extract_pooled('image.jpg')")
    print(f"  >>> # features.shape = (1, {features.shape[1]})")
    print(f"  >>> ")
    print(f"  >>> # Option 1: Use with sklearn")
    print(f"  >>> from sklearn.linear_model import LogisticRegression")
    print(f"  >>> classifier = LogisticRegression()")
    print(f"  >>> classifier.fit(features_train, labels_train)")
    print(f"  >>> ")
    print(f"  >>> # Option 2: Use with PyTorch")
    print(f"  >>> import torch.nn as nn")
    print(f"  >>> classifier = nn.Sequential(")
    print(f"  ...     nn.Linear({features.shape[1]}, 512),")
    print(f"  ...     nn.ReLU(),")
    print(f"  ...     nn.Dropout(0.3),")
    print(f"  ...     nn.Linear(512, num_classes)")
    print(f"  ... )")
    
    # Show size comparison
    print(f"\n[Step 3] Storage efficiency")
    latent = extractor.extract_from_image(test_image)
    pooled = features
    
    latent_size = latent.element_size() * latent.nelement() / (1024**2)
    pooled_size = pooled.element_size() * pooled.nelement() / (1024**2)
    
    print(f"  Full latent:     {latent_size:.2f} MB")
    print(f"  Pooled features: {pooled_size:.4f} MB")
    print(f"  Compression:     {latent_size/pooled_size:.0f}x smaller")
    
    print(f"\n[Step 4] Advantages over diffusion-based approach")
    print(f"  ✓ No text encoder needed")
    print(f"  ✓ No diffusion sampling (35+ steps)")
    print(f"  ✓ Deterministic (same input → same features)")
    print(f"  ✓ Much faster (~10-50x depending on config)")
    print(f"  ✓ Lower memory usage")
    print(f"  ✓ Perfect for supervised learning tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Test Image2Representation extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='2B/post-trained',
                       help='Model name/size')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints',
                       help='Path to model checkpoint (use "checkpoints" to auto-download from HuggingFace)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name (optional, will use model default if not specified)')
    
    # Input arguments
    parser.add_argument('--image', type=str,
                       help='Path to test image')
    parser.add_argument('--video', type=str,
                       help='Path to test video')
    
    # Test modes
    parser.add_argument('--benchmark', action='store_true',
                       help='Run speed benchmarks')
    parser.add_argument('--batch_test', action='store_true',
                       help='Test batch processing')
    parser.add_argument('--demo_classifier', action='store_true',
                       help='Demo classifier training pipeline')
    parser.add_argument('--num_iterations', type=int, default=10,
                       help='Number of iterations for benchmarks')
    
    # Other
    parser.add_argument('--resolution', type=str, default='360,640',
                       help='Target resolution (H,W)')
    
    args = parser.parse_args()
    
    # Parse resolution
    target_h, target_w = map(int, args.resolution.split(','))
    resolution = (target_h, target_w)
    
    print("="*80)
    print("IMAGE2REPRESENTATION EXTRACTION TEST")
    print("="*80)
    print(f"\nModel:      {args.model}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Experiment: {args.experiment}")
    print(f"Resolution: {target_h}x{target_w}")
    
    # Initialize extractor
    print(f"\n{'='*80}")
    print("Initializing extractor...")
    print(f"{'='*80}")
    
    start_init = time.time()
    
    # Ignore placeholder experiment names
    experiment = args.experiment
    if experiment and 'your-experiment' in experiment.lower():
        print("⚠️  Ignoring placeholder experiment name, using auto-detect")
        experiment = None
    
    extractor = Image2RepresentationExtractor(
        experiment_name=experiment,
        ckpt_path=args.ckpt_path if args.ckpt_path != 'checkpoints' else None,
        model_size=args.model.split('/')[0],  # Extract "2B" or "14B" from "2B/post-trained"
    )
    init_time = time.time() - start_init
    
    print(f"\n✓ Initialization complete in {format_time(init_time)}")
    
    # Get latent shape info
    print(f"\n{'='*80}")
    print("Model Information")
    print(f"{'='*80}")
    print(f"Required input frames: {extractor.model_required_frames}")
    latent_shape = extractor.get_latent_shape()
    print(f"Latent shape (C,T,H,W): {latent_shape}")
    print(f"Latent dimensions: {np.prod(latent_shape):,} values")
    
    # Determine test input
    test_input = args.image or args.video
    if not test_input:
        print("\n⚠️  No test input provided. Use --image or --video to specify input.")
        print("Skipping tests...")
        return
    
    if not Path(test_input).exists():
        print(f"\n❌ Error: Input file not found: {test_input}")
        return
    
    # Run tests
    try:
        if args.image:
            results = test_single_image(extractor, args.image)
        elif args.video:
            results = test_single_video(extractor, args.video)
        
        if args.benchmark:
            benchmark_speed(extractor, test_input, args.num_iterations)
        
        if args.batch_test:
            test_batch_processing(extractor, test_input)
        
        if args.demo_classifier:
            test_classifier_pipeline(extractor, test_input)
    
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ ALL TESTS COMPLETE")
    print(f"{'='*80}")
    
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Peak:      {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Extract features from your dataset:")
    print("   python examples/extract_features_for_classifier.py \\")
    print("       --experiment your-exp \\")
    print("       --ckpt_path path/to/ckpt \\")
    print("       --input_dir path/to/images \\")
    print("       --output_file features.pt")
    print("\n2. Train your classifier:")
    print("   >>> data = torch.load('features.pt')")
    print("   >>> features = data['features']")
    print("   >>> # Train sklearn/PyTorch classifier")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
