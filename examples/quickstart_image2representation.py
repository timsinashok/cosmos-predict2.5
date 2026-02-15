#!/usr/bin/env python3
"""
Quick start script - Extract features from a single image/video.

This is the simplest way to get started with feature extraction.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmos_predict2._src.predict2.inference.image2representation import Image2RepresentationExtractor
import torch


def main():
    print("="*80)
    print("QUICK START: Image2Representation Feature Extraction")
    print("="*80)
    
    # Configuration - UPDATE THESE VALUES
    EXPERIMENT_NAME = "your-experiment-name"  # TODO: Update this
    CHECKPOINT_PATH = "path/to/checkpoint"     # TODO: Update this
    TEST_IMAGE = "path/to/test/image.jpg"     # TODO: Update this
    
    print("\n⚠️  Before running, please update the configuration in this script:")
    print(f"   EXPERIMENT_NAME = '{EXPERIMENT_NAME}'")
    print(f"   CHECKPOINT_PATH = '{CHECKPOINT_PATH}'")
    print(f"   TEST_IMAGE = '{TEST_IMAGE}'")
    
    # Check if using defaults
    if EXPERIMENT_NAME == "your-experiment-name":
        print("\n❌ Please update EXPERIMENT_NAME in the script!")
        print("   Edit: examples/quickstart_image2representation.py")
        return
    
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    if not Path(TEST_IMAGE).exists():
        print(f"\n❌ Test image not found: {TEST_IMAGE}")
        return
    
    # Initialize extractor
    print(f"\n{'='*80}")
    print("Step 1: Initializing extractor...")
    print(f"{'='*80}")
    
    extractor = Image2RepresentationExtractor(
        experiment_name=EXPERIMENT_NAME,
        ckpt_path=CHECKPOINT_PATH,
    )
    
    print("\n✓ Extractor ready!")
    
    # Extract features
    print(f"\n{'='*80}")
    print("Step 2: Extracting features...")
    print(f"{'='*80}")
    
    # Option 1: Full latent representation
    print("\n[Option 1] Full latent representation:")
    latent = extractor.extract_from_image(TEST_IMAGE)
    print(f"  Shape: {tuple(latent.shape)}")
    print(f"  Size:  {latent.element_size() * latent.nelement() / (1024**2):.2f} MB")
    
    # Option 2: Pooled feature vector (recommended for classifiers)
    print("\n[Option 2] Pooled feature vector (mean):")
    features = extractor.extract_pooled(TEST_IMAGE, pool_method="mean")
    print(f"  Shape: {tuple(features.shape)}")
    print(f"  Size:  {features.element_size() * features.nelement() / (1024):.2f} KB")
    
    # Save features
    print(f"\n{'='*80}")
    print("Step 3: Saving features...")
    print(f"{'='*80}")
    
    output_file = Path("extracted_features.pt")
    torch.save({
        'latent': latent,
        'features': features,
        'image_path': TEST_IMAGE,
    }, output_file)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Next steps
    print(f"\n{'='*80}")
    print("Next Steps:")
    print(f"{'='*80}")
    print("\n1. To extract from multiple images:")
    print("   python examples/extract_features_for_classifier.py \\")
    print("       --experiment your-exp \\")
    print("       --ckpt_path path/to/ckpt \\")
    print("       --input_dir path/to/images \\")
    print("       --output_file all_features.pt")
    
    print("\n2. To train a classifier:")
    print("   >>> data = torch.load('all_features.pt')")
    print("   >>> features = data['features']")
    print("   >>> # Train your classifier here")
    
    print("\n3. For comprehensive testing:")
    print("   python cosmos-predict2.5_old_messy/examples/test_embedding_generation.py \\")
    print("       --experiment your-exp \\")
    print("       --ckpt_path path/to/ckpt \\")
    print("       --image test.jpg \\")
    print("       --benchmark --demo_classifier")
    
    print(f"\n{'='*80}")
    print("✓ QUICK START COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
