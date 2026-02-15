# Architecture Comparison

## Traditional Video2World Pipeline (with Diffusion)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FULL PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input Image/Video                                                        │
│         │                                                                 │
│         ▼                                                                 │
│  ┌──────────────┐                                                        │
│  │   Tokenizer  │                                                        │
│  │   Encoder    │ ←── Loads ~2-4 GB GPU memory                           │
│  └──────┬───────┘                                                        │
│         │                                                                 │
│         ▼                                                                 │
│  Latent Representation                                                    │
│  (C, T, H, W)                                                            │
│         │                                                                 │
│         ▼                                                                 │
│  ┌──────────────┐      ┌──────────────┐                                 │
│  │  Text        │      │  Diffusion   │                                  │
│  │  Encoder     │◄────►│  Model       │ ←── Loads ~8-12 GB GPU memory   │
│  │  (T5-11B)    │      │  (DiT)       │                                  │
│  └──────────────┘      └──────┬───────┘                                 │
│         │                      │                                          │
│         │      35+ Sampling Steps                                        │
│         │      (SLOW: 5-30s)                                             │
│         │                      │                                          │
│         └──────────┬───────────┘                                         │
│                    ▼                                                      │
│             Denoised Latent                                              │
│                    │                                                      │
│                    ▼                                                      │
│             ┌──────────────┐                                             │
│             │  Tokenizer   │                                             │
│             │  Decoder     │ ←── Loads ~2-4 GB GPU memory                │
│             └──────┬───────┘                                             │
│                    │                                                      │
│                    ▼                                                      │
│             Generated Video                                              │
│                                                                           │
│  Total: ~15-20 GB GPU | 5-30s per image                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## New Image2Representation Pipeline (NO Diffusion)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input Image/Video                                                        │
│         │                                                                 │
│         ▼                                                                 │
│  ┌──────────────┐                                                        │
│  │   Tokenizer  │                                                        │
│  │   Encoder    │ ←── Loads ~2-4 GB GPU memory                           │
│  └──────┬───────┘                                                        │
│         │                                                                 │
│         ▼                                                                 │
│  Latent Representation                                                    │
│  (C, T, H, W)                                                            │
│         │                                                                 │
│         ▼                                                                 │
│  ┌──────────────┐                                                        │
│  │   Pooling    │                                                        │
│  │   (Optional) │                                                        │
│  └──────┬───────┘                                                        │
│         │                                                                 │
│         ▼                                                                 │
│  Feature Vector (C,)                                                     │
│         │                                                                 │
│         ▼                                                                 │
│  Your Classifier                                                         │
│  (sklearn/PyTorch)                                                       │
│         │                                                                 │
│         ▼                                                                 │
│  Prediction                                                              │
│                                                                           │
│  Total: ~2-4 GB GPU | 0.1-0.5s per image                                │
│  Speedup: 10-50x faster! 🚀                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## What's Different?

### Removed Components

| Component | Memory | Purpose | Why Removed |
|-----------|--------|---------|-------------|
| Text Encoder (T5-11B) | ~4-6 GB | Text conditioning | Not needed for feature extraction |
| Diffusion Model (DiT) | ~8-12 GB | Video generation | Not generating, just extracting |
| Decoder | ~2-4 GB | Latent → Video | Not generating output video |
| Sampling Loop | N/A | 35+ denoising steps | Not generating, just encoding |

### Kept Components

| Component | Memory | Purpose | Why Kept |
|-----------|--------|---------|----------|
| Tokenizer Encoder | ~2-4 GB | Image → Latent | This IS the feature extractor |

## Memory Comparison

```
Full Pipeline:     ████████████████████ 15-20 GB
                   │
                   │
Image2Repr:        ████                  2-4 GB
                   
Savings:           75-80% less memory! 💾
```

## Speed Comparison

```
Full Pipeline:     ██████████████████████████████ 5-30s per image
                   
Image2Repr:        █                               0.1-0.5s per image
                   
Speedup:           10-50x faster! ⚡
```

## Use Case Comparison

### Full Pipeline (Video2World)
✅ Generating new videos  
✅ Text-guided video synthesis  
✅ Creative applications  
✅ Conditional generation  

❌ Training classifiers (overkill)  
❌ Feature extraction (too slow)  
❌ Batch processing (memory intensive)  

### Image2Representation
✅ Training classifiers on video features  
✅ Fast feature extraction  
✅ Deterministic representations  
✅ Batch processing  
✅ Video similarity search  
✅ Action recognition  

❌ Generating videos (not designed for this)  
❌ Text-guided generation (no text encoder)  

## Example: Collision Detection Workflow

### Traditional Approach (SLOW)
```
1. Load full model      → 15-20 GB GPU
2. For each video:
   - Text encode        → 0.5s
   - Diffusion sample   → 5-30s (❌ not even needed!)
   - Decode             → 1-2s
3. Extract features     → from generated video?
4. Train classifier     → on these features
```

### Image2Representation Approach (FAST)
```
1. Load encoder only    → 2-4 GB GPU
2. For each video:
   - Encode to latent   → 0.1-0.5s ✅
   - Pool to vector     → 0.001s ✅
3. Train classifier     → on feature vectors
4. Deploy              → fast inference!
```

## Performance Numbers (Example)

For 1000 videos:

| Pipeline | Time | GPU Memory | Storage |
|----------|------|------------|---------|
| Full Pipeline | ~2-8 hours | 15-20 GB | ~50-100 GB |
| Image2Repr | ~10-30 min | 2-4 GB | ~5-10 MB |

**Savings:** ~10-16x faster, ~75% less memory, ~1000x less storage!

## Code Comparison

### Full Pipeline
```python
from cosmos_predict2.inference import Inference

# Loads everything
inference = Inference(setup_args)

# Generate video (slow)
video = inference.generate_vid2world(
    prompt="a video",  # Still need text!
    input_path="image.jpg",
    num_steps=35,      # 35 sampling steps
)

# Now what? Extract features from generated video?
```

### Image2Representation
```python
from cosmos_predict2._src.predict2.inference.image2representation import Image2RepresentationExtractor

# Loads only encoder
extractor = Image2RepresentationExtractor(
    experiment_name="exp",
    ckpt_path="ckpt"
)

# Extract features (fast)
features = extractor.extract_pooled("image.jpg")
# Done! Ready for classifier training
```

## Architecture Details

### What the Encoder Does
```
Input Image (H, W, 3)
        ↓
   [Conv layers]
        ↓
   [Attention blocks]
        ↓
   [Downsampling]
        ↓
Latent (C, T, H', W')
   where H' << H, W' << W
```

The encoder compresses the image into a compact latent representation that captures:
- Visual features
- Semantic information
- Motion patterns (for videos)
- Spatial structure

### Pooling Options

**Mean Pooling:**
```
Latent (1, C, T, H, W)
        ↓
   mean(dim=[2,3,4])
        ↓
Features (1, C)
```
Good for: Overall scene understanding

**Max Pooling:**
```
Latent (1, C, T, H, W)
        ↓
   max(dim=[2,3,4])
        ↓
Features (1, C)
```
Good for: Detecting specific features

**No Pooling:**
```
Latent (1, C, T, H, W)
        ↓
Keep as is
        ↓
Latent (1, C, T, H, W)
```
Good for: Tasks needing spatial information

## Summary

| Aspect | Full Pipeline | Image2Representation |
|--------|---------------|---------------------|
| **Purpose** | Video generation | Feature extraction |
| **Speed** | 5-30s | 0.1-0.5s |
| **Memory** | 15-20 GB | 2-4 GB |
| **Text Encoder** | ✅ Required | ❌ Not needed |
| **Diffusion** | ✅ 35+ steps | ❌ Skipped |
| **Deterministic** | ❌ Random | ✅ Yes |
| **Use Case** | Generation | Classification |
| **Output** | Video | Features |

**When to use what:**
- **Full Pipeline:** When you want to *generate* new videos
- **Image2Representation:** When you want to *analyze* videos with classifiers
