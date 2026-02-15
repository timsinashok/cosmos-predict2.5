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
Image to Representation Extraction (NO DIFFUSION)

This module extracts latent representations from images/videos using only the
tokenizer encoder, bypassing the diffusion model entirely.

Use case: Feature extraction for training classifiers on video representations.

Example usage:
    encoder = Image2RepresentationExtractor(
        experiment_name="your-experiment",
        ckpt_path="path/to/checkpoint"
    )
    
    features = encoder.extract_from_image("path/to/image.jpg")
    # features shape: (1, C, T, H, W) - latent representation
    
    # Optional: pool to fixed-size vector
    pooled = encoder.extract_pooled("path/to/image.jpg")
    # pooled shape: (1, C) - single vector per image
"""

import os
import time
from pathlib import Path
from typing import Literal

import torch
import torchvision
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.inference.video2world import resize_input
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint


class Image2RepresentationExtractor:
    """
    Extracts latent representations from images using only the tokenizer encoder.
    
    This class loads a trained Cosmos model but ONLY uses the encoder portion,
    skipping the diffusion model, text encoder, and decoder entirely.
    
    Perfect for:
    - Feature extraction for downstream classifiers
    - Fast representation learning
    - Training on video representations without generation
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        ckpt_path: str | None = None,
        s3_credential_path: str = "",
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
        experiment_opts: list[str] | None = None,
        device: str = "cuda",
        model_size: str = "2B",
    ):
        """
        Initialize the feature extractor.
        
        Args:
            experiment_name: Name of the experiment configuration (None for auto-detect)
            ckpt_path: Path to model checkpoint (None for auto-download from HuggingFace)
            s3_credential_path: Path to S3 credentials if needed
            config_file: Path to config file
            experiment_opts: Additional config overrides
            device: Device to run on ('cuda' or 'cpu')
            model_size: Model size ("2B" or "14B") used for auto-detecting experiment
        """
        self.s3_credential_path = s3_credential_path
        self.device = device
        
        # Auto-detect experiment name if not provided
        if experiment_name is None:
            if '14B' in model_size.upper():
                experiment_name = 'Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40'
            else:
                experiment_name = 'Stage-c_pt_4-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22'
            log.info(f"Auto-detected experiment: {experiment_name}")
        
        self.experiment_name = experiment_name
        
        # If no checkpoint path, use HuggingFace default
        if ckpt_path is None or ckpt_path == "checkpoints":
            # Use HuggingFace with proper URI format
            if '14B' in model_size.upper():
                ckpt_path = "hf://nvidia/Cosmos-Predict2.5-14B/base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt"
            else:
                ckpt_path = "hf://nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
            log.info(f"Using HuggingFace model: {ckpt_path}")
        
        self.ckpt_path = ckpt_path
        
        log.info("="*80)
        log.info("INITIALIZING IMAGE2REPRESENTATION EXTRACTOR")
        log.info("="*80)
        log.info("This will load ONLY the tokenizer encoder (NO diffusion, NO text encoder)")
        
        # Load the full model first
        if experiment_opts is None:
            experiment_opts = []
        if not INTERNAL:
            experiment_opts.append("~data_train")
        
        log.info(f"Loading model from: {ckpt_path}")
        model, config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            s3_checkpoint_dir=ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
            to_device=device,
        )
        
        # Extract and keep ONLY the tokenizer for encoding
        # The tokenizer has an encode() method, not a separate encoder attribute
        self.tokenizer = model.tokenizer
        # Set to eval mode on the underlying model if it exists
        if hasattr(self.tokenizer, 'model') and hasattr(self.tokenizer.model, 'eval'):
            self.tokenizer.model.eval()
        
        # Store model config (needed for state_t)
        self.model_config = model.config
        self.config = config
        
        # Get model requirements
        self.model_required_frames = self.tokenizer.get_pixel_num_frames(self.model_config.state_t)
        log.info(f"Model requires {self.model_required_frames} frames for encoding")
        
        # Clean up - we don't need the diffusion model, text encoder, or decoder
        if hasattr(model, 'net'):
            del model.net
        if hasattr(model, 'text_encoder'):
            del model.text_encoder
        if hasattr(model, 'conditioner'):
            del model.conditioner
        del model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log.info("✓ Encoder loaded successfully")
        log.info("✓ Diffusion model NOT loaded (memory saved)")
        log.info("✓ Text encoder NOT loaded (memory saved)")
        log.info("✓ Decoder NOT loaded (memory saved)")
        log.info("="*80)
        
    def preprocess_image(
        self,
        image_path: str | Path,
        resolution: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Load and preprocess an image for encoding.
        
        Args:
            image_path: Path to image file
            resolution: Target (H, W) resolution, or None for default (360, 640)
            
        Returns:
            Tensor of shape (1, C, T, H, W) ready for encoding
        """
        if resolution is None:
            # Use default resolution
            resolution = (360, 640)
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Convert to tensor [0, 1] range
        img = torchvision.transforms.functional.to_tensor(img)  # (C, H, W)
        
        # Add temporal dimension
        img = img.unsqueeze(0)  # (T=1, C, H, W)
        
        # Convert to uint8 range for resize_input
        img = (img * 255.0).to(torch.uint8)
        
        # Resize if needed
        img = resize_input(img, list(resolution))
        
        # Create video tensor with required number of frames
        # First frame is the image, rest are zeros
        num_frames = self.model_required_frames
        if num_frames > 1:
            padding = torch.zeros(num_frames - 1, img.shape[1], img.shape[2], img.shape[3], dtype=torch.uint8)
            img = torch.cat([img, padding], dim=0)
        
        # Rearrange to (B, C, T, H, W) and keep as uint8
        img = img.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        
        return img
    
    def preprocess_video(
        self,
        video_path: str | Path,
        resolution: tuple[int, int] | None = None,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        """
        Load and preprocess a video for encoding.
        
        Args:
            video_path: Path to video file
            resolution: Target (H, W) resolution, or None for default (360, 640)
            num_frames: Number of frames to use (default: model requirement)
            
        Returns:
            Tensor of shape (1, C, T, H, W) ready for encoding (uint8)
        """
        from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
        
        if resolution is None:
            # Use default resolution
            resolution = (360, 640)
        
        if num_frames is None:
            num_frames = self.model_required_frames
        
        # Load video
        video_frames, _ = easy_io.load(str(video_path))
        video_tensor = torch.from_numpy(video_frames)  # uint8 already
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Take last N frames or pad
        available_frames = video_tensor.shape[1]
        if available_frames > num_frames:
            video_tensor = video_tensor[:, -num_frames:, :, :]
        elif available_frames < num_frames:
            padding = video_tensor[:, -1:, :, :].repeat(1, num_frames - available_frames, 1, 1)
            video_tensor = torch.cat([video_tensor, padding], dim=1)
        
        # Resize
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)
        video_tensor = resize_input(video_tensor, list(resolution))
        
        # Rearrange to (B, C, T, H, W) and keep as uint8
        video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        
        return video_tensor
    
    @torch.inference_mode()
    def extract_latent(
        self,
        input_tensor: torch.Tensor,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Extract latent representation from preprocessed tensor.
        
        Args:
            input_tensor: Preprocessed tensor (B, C, T, H, W) in uint8 [0, 255]
            return_cpu: Move result to CPU before returning
            
        Returns:
            Latent representation tensor of shape (B, C_latent, T_latent, H_latent, W_latent)
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Convert to float and normalize to [-1, 1] range
        if input_tensor.dtype == torch.uint8:
            input_tensor = input_tensor.float() / 255.0  # [0, 1]
        input_tensor = input_tensor * 2.0 - 1.0  # [-1, 1]
        
        # Convert to bfloat16 for efficiency (matches training)
        input_tensor = input_tensor.to(torch.bfloat16)
        
        # Encode using tokenizer's encode method
        latent = self.tokenizer.encode(input_tensor)
        
        # Move to CPU if requested
        if return_cpu:
            latent = latent.cpu()
        
        return latent
    
    @torch.inference_mode()
    def extract_from_image(
        self,
        image_path: str | Path,
        resolution: tuple[int, int] | None = None,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Extract latent representation from an image file.
        
        Args:
            image_path: Path to image file
            resolution: Target (H, W) resolution
            return_cpu: Move result to CPU
            
        Returns:
            Latent representation tensor
        """
        img_tensor = self.preprocess_image(image_path, resolution)
        return self.extract_latent(img_tensor, return_cpu)
    
    @torch.inference_mode()
    def extract_from_video(
        self,
        video_path: str | Path,
        resolution: tuple[int, int] | None = None,
        num_frames: int | None = None,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Extract latent representation from a video file.
        
        Args:
            video_path: Path to video file
            resolution: Target (H, W) resolution
            num_frames: Number of frames to use
            return_cpu: Move result to CPU
            
        Returns:
            Latent representation tensor
        """
        vid_tensor = self.preprocess_video(video_path, resolution, num_frames)
        return self.extract_latent(vid_tensor, return_cpu)
    
    @torch.inference_mode()
    def extract_pooled(
        self,
        input_path: str | Path,
        resolution: tuple[int, int] | None = None,
        pool_method: Literal["mean", "max", "flatten"] = "mean",
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Extract and pool latent representation to fixed-size vector.
        
        Args:
            input_path: Path to image or video file
            resolution: Target resolution
            pool_method: How to pool spatial-temporal dimensions:
                - "mean": Average pool over T, H, W
                - "max": Max pool over T, H, W
                - "flatten": Flatten to 1D vector (large!)
            return_cpu: Move result to CPU
            
        Returns:
            Pooled feature vector
        """
        # Determine if input is image or video
        ext = Path(input_path).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp']:
            latent = self.extract_from_image(input_path, resolution, return_cpu=False)
        else:
            latent = self.extract_from_video(input_path, resolution, return_cpu=False)
        
        # Pool spatial-temporal dimensions
        # latent shape: (B, C, T, H, W)
        if pool_method == "mean":
            pooled = latent.mean(dim=[2, 3, 4])  # (B, C)
        elif pool_method == "max":
            pooled = latent.amax(dim=[2, 3, 4])  # (B, C)
        elif pool_method == "flatten":
            pooled = latent.flatten(start_dim=1)  # (B, C*T*H*W)
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")
        
        if return_cpu:
            pooled = pooled.cpu()
        
        return pooled
    
    @torch.inference_mode()
    def extract_batch(
        self,
        input_paths: list[str | Path],
        resolution: tuple[int, int] | None = None,
        batch_size: int = 4,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Extract latent representations from multiple inputs in batches.
        
        Args:
            input_paths: List of paths to image/video files
            resolution: Target resolution
            batch_size: Number of inputs to process at once
            return_cpu: Move results to CPU
            
        Returns:
            Stacked latent representations (N, C_latent, T_latent, H_latent, W_latent)
        """
        all_latents = []
        
        for i in range(0, len(input_paths), batch_size):
            batch_paths = input_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for path in batch_paths:
                ext = Path(path).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    tensor = self.preprocess_image(path, resolution)
                else:
                    tensor = self.preprocess_video(path, resolution)
                batch_tensors.append(tensor)
            
            # Stack and encode
            batch = torch.cat(batch_tensors, dim=0)
            latents = self.extract_latent(batch, return_cpu=False)
            
            if return_cpu:
                latents = latents.cpu()
            
            all_latents.append(latents)
        
        # Concatenate all batches
        return torch.cat(all_latents, dim=0)
    
    def get_latent_shape(self) -> tuple:
        """
        Get the shape of latent representations (without batch dimension).
        
        Returns:
            Tuple of (C_latent, T_latent, H_latent, W_latent)
        """
        # Create dummy input in [-1, 1] range
        dummy = torch.zeros(1, 3, self.model_required_frames, 360, 640).to(self.device) * 2.0 - 1.0
        with torch.inference_mode():
            latent = self.tokenizer.encode(dummy)
        return tuple(latent.shape[1:])  # Skip batch dimension
