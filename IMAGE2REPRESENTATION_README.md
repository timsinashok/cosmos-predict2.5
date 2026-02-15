# Image2Representation: Feature Extraction for Classifier Training

This module provides a fast, efficient way to extract latent representations from images/videos using **only the encoder** (no diffusion model, no text encoder).

## Key Advantages

✅ **No text encoder needed** - Pure image/video features  
✅ **No diffusion sampling** - Direct encoding (35+ steps saved)  
✅ **Deterministic** - Same input → same features every time  
✅ **Fast** - 10-50x faster than full diffusion pipeline  
✅ **Memory efficient** - Only loads the encoder  
✅ **Perfect for supervised learning** - Train classifiers on these features

## Files

1. **`cosmos_predict2/_src/predict2/inference/image2representation.py`**
   - Core `Image2RepresentationExtractor` class
   - Handles image/video preprocessing and encoding
   - Provides pooling options for fixed-size vectors

2. **`examples/extract_features_for_classifier.py`**
   - Batch extract features from a directory
   - Save features for training

3. **`examples/test_embedding_generation.py`** (in old_messy folder)
   - Comprehensive test suite
   - Benchmarking tools
   - Demo of classifier training pipeline

## Quick Start

### 1. Extract Features from a Single Image

```python
from cosmos_predict2._src.predict2.inference.image2representation import Image2RepresentationExtractor

# Initialize extractor
extractor = Image2RepresentationExtractor(
    experiment_name="your-experiment",
    ckpt_path="path/to/checkpoint"
)

# Extract full latent representation
latent = extractor.extract_from_image("image.jpg")
# Shape: (1, C, T, H, W) - spatial-temporal latent

# Or extract pooled feature vector
features = extractor.extract_pooled("image.jpg", pool_method="mean")
# Shape: (1, C) - single vector per image
```

### 2. Extract Features from a Dataset

```bash
python examples/extract_features_for_classifier.py \
    --experiment your-experiment \
    --ckpt_path path/to/checkpoint \
    --input_dir path/to/images \
    --output_file features.pt \
    --batch_size 4 \
    --resolution 360,640
```

### 3. Train a Classifier

```python
import torch
from sklearn.linear_model import LogisticRegression

# Load extracted features
data = torch.load('features.pt')
features = data['features'].numpy()  # (N, C)
labels = your_labels  # (N,)

# Train classifier
classifier = LogisticRegression()
classifier.fit(features, labels)

# Predict on new image
new_features = extractor.extract_pooled("new_image.jpg")
prediction = classifier.predict(new_features.numpy())
```

## Testing & Benchmarking

Run the comprehensive test suite:

```bash
# Test single image extraction
python cosmos-predict2.5_old_messy/examples/test_embedding_generation.py \
    --experiment your-experiment \
    --ckpt_path path/to/checkpoint \
    --image test_image.jpg

# Run speed benchmarks
python cosmos-predict2.5_old_messy/examples/test_embedding_generation.py \
    --experiment your-experiment \
    --ckpt_path path/to/checkpoint \
    --image test_image.jpg \
    --benchmark \
    --num_iterations 20

# Test batch processing
python cosmos-predict2.5_old_messy/examples/test_embedding_generation.py \
    --experiment your-experiment \
    --ckpt_path path/to/checkpoint \
    --image test_image.jpg \
    --batch_test

# Demo classifier training pipeline
python cosmos-predict2.5_old_messy/examples/test_embedding_generation.py \
    --experiment your-experiment \
    --ckpt_path path/to/checkpoint \
    --image test_image.jpg \
    --demo_classifier
```

## API Reference

### `Image2RepresentationExtractor`

#### Initialization

```python
extractor = Image2RepresentationExtractor(
    experiment_name: str,        # Experiment configuration name
    ckpt_path: str,              # Path to model checkpoint
    s3_credential_path: str = "", # S3 credentials if needed
    config_file: str = "...",    # Path to config file
    device: str = "cuda"         # Device to use
)
```

#### Methods

**`extract_from_image(image_path, resolution=None, return_cpu=True)`**
- Extract full latent representation from image
- Returns: `(1, C_latent, T_latent, H_latent, W_latent)`

**`extract_from_video(video_path, resolution=None, num_frames=None, return_cpu=True)`**
- Extract full latent representation from video
- Returns: `(1, C_latent, T_latent, H_latent, W_latent)`

**`extract_pooled(input_path, resolution=None, pool_method='mean', return_cpu=True)`**
- Extract and pool to fixed-size vector
- `pool_method`: "mean", "max", or "flatten"
- Returns: `(1, C)` for mean/max, `(1, C*T*H*W)` for flatten

**`extract_batch(input_paths, resolution=None, batch_size=4, return_cpu=True)`**
- Extract features from multiple inputs
- Returns: `(N, C_latent, T_latent, H_latent, W_latent)`

**`get_latent_shape()`**
- Get shape of latent representations
- Returns: `(C_latent, T_latent, H_latent, W_latent)`

## Pooling Methods

### Mean Pooling (Recommended)
```python
features = extractor.extract_pooled(image_path, pool_method="mean")
# Averages over spatial-temporal dimensions
# Shape: (1, C) - compact vector
```

### Max Pooling
```python
features = extractor.extract_pooled(image_path, pool_method="max")
# Takes maximum over spatial-temporal dimensions
# Shape: (1, C) - captures peak activations
```

### Flatten (Use with caution)
```python
features = extractor.extract_pooled(image_path, pool_method="flatten")
# Flattens all dimensions
# Shape: (1, C*T*H*W) - very large vector!
```

## Use Cases

### 1. Video Collision Detection
```python
# Extract features from collision/normal videos
features_collision = [extractor.extract_pooled(v) for v in collision_videos]
features_normal = [extractor.extract_pooled(v) for v in normal_videos]

# Train binary classifier
X = torch.cat(features_collision + features_normal)
y = [1]*len(features_collision) + [0]*len(features_normal)

classifier.fit(X.numpy(), y)
```

### 2. Action Recognition
```python
# Extract features from videos of different actions
features_by_action = {}
for action in actions:
    features = [extractor.extract_pooled(v) for v in action_videos[action]]
    features_by_action[action] = features

# Train multi-class classifier
```

### 3. Video Similarity Search
```python
# Extract features from database
db_features = [extractor.extract_pooled(v) for v in database_videos]

# Query with new video
query_features = extractor.extract_pooled(query_video)

# Find most similar (cosine similarity)
similarities = torch.nn.functional.cosine_similarity(
    query_features, 
    torch.cat(db_features)
)
```

## Performance Tips

1. **Use batch processing** for multiple images:
   ```python
   features = extractor.extract_batch(image_paths, batch_size=8)
   ```

2. **Use pooled features** if you don't need spatial information:
   ```python
   features = extractor.extract_pooled(image_path)  # Much smaller
   ```

3. **Adjust resolution** based on your needs:
   ```python
   # Lower resolution = faster + less memory
   features = extractor.extract_from_image(path, resolution=(180, 320))
   ```

## Comparison with Full Pipeline

| Feature | Image2Representation | Full Diffusion Pipeline |
|---------|---------------------|-------------------------|
| Speed | ~0.1-0.5s per image | ~5-30s per image |
| Memory | ~2-4 GB GPU | ~10-20 GB GPU |
| Text encoder | ❌ Not needed | ✅ Required |
| Deterministic | ✅ Yes | ❌ No (sampling) |
| Use case | Feature extraction | Video generation |
| Output | Latent features | Generated video |

## Troubleshooting

### Out of Memory
- Reduce batch size: `batch_size=1`
- Lower resolution: `resolution=(180, 320)`
- Use pooled extraction: `extract_pooled()` instead of `extract_from_image()`

### Slow Processing
- Increase batch size if memory allows
- Ensure GPU is being used: `device="cuda"`
- Use lower resolution for preprocessing

### Different Feature Sizes
- Use `pool_method="mean"` or `pool_method="max"` for fixed-size vectors
- These are compatible with sklearn and most classifiers

## Next Steps

1. Extract features from your dataset
2. Save features: `torch.save(features, 'features.pt')`
3. Train classifier on features
4. Evaluate on held-out test set
5. Deploy for inference

## Questions?

See the test script for comprehensive examples:
```bash
python cosmos-predict2.5_old_messy/examples/test_embedding_generation.py --help
```
