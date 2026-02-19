"""
Configuration Tuning Guide for People Counting System

This guide provides recommended parameters based on:
- Scene density (sparse, medium, crowded)
- Camera setup (framerate, resolution, viewing angle)
- GPU capability
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SceneConfig:
    """Scene-specific configuration"""
    # Scene characteristics
    max_people_per_frame: int
    fps: int
    resolution: Tuple[int, int]  # (width, height)
    
    # Detection parameters
    detection_confidence: float
    nms_threshold: float
    
    # Tracking parameters
    max_disappeared: int  # frames
    min_hits_confirm: int
    
    # Matching parameters - Stage 1 (IoU)
    iou_gate: float
    stage1_threshold: float
    
    # Matching parameters - Stage 2 (Appearance)
    spatial_gate: float  # normalized (0-1)
    appearance_gate: float  # cosine similarity
    stage2_threshold: float
    spatial_weight: float
    appearance_weight: float
    
    # Feature parameters
    feature_ema_alpha: float
    feature_history_size: int
    
    # Re-entry parameters
    reentry_similarity_threshold: float
    reentry_spatial_gate: int  # pixels
    
    # Exit signature parameters
    exit_best_n_frames: int


# Preset configurations for different scenarios

SPARSE_SCENE = SceneConfig(
    max_people_per_frame=5,
    fps=30,
    resolution=(1920, 1080),
    
    detection_confidence=0.5,
    nms_threshold=0.4,
    
    max_disappeared=45,  # 1.5 seconds at 30fps
    min_hits_confirm=2,  # Quick confirmation
    
    iou_gate=0.2,  # Relaxed - people move more freely
    stage1_threshold=0.8,
    
    spatial_gate=0.6,  # Relaxed spatial constraint
    appearance_gate=0.25,  # Lower threshold
    stage2_threshold=0.65,
    spatial_weight=0.3,
    appearance_weight=0.7,
    
    feature_ema_alpha=0.85,  # Less smoothing (faster adaptation)
    feature_history_size=20,
    
    reentry_similarity_threshold=0.45,
    reentry_spatial_gate=250,
    
    exit_best_n_frames=10
)


MEDIUM_SCENE = SceneConfig(
    max_people_per_frame=15,
    fps=30,
    resolution=(1920, 1080),
    
    detection_confidence=0.55,
    nms_threshold=0.45,
    
    max_disappeared=30,  # 1 second at 30fps
    min_hits_confirm=3,  # Standard confirmation
    
    iou_gate=0.3,  # Standard gate
    stage1_threshold=0.7,
    
    spatial_gate=0.5,  # Medium constraint
    appearance_gate=0.3,  # Standard threshold
    stage2_threshold=0.6,
    spatial_weight=0.3,
    appearance_weight=0.7,
    
    feature_ema_alpha=0.9,  # Standard smoothing
    feature_history_size=30,
    
    reentry_similarity_threshold=0.5,
    reentry_spatial_gate=200,
    
    exit_best_n_frames=15
)


CROWDED_SCENE = SceneConfig(
    max_people_per_frame=50,
    fps=30,
    resolution=(1920, 1080),
    
    detection_confidence=0.6,  # Higher to reduce false positives
    nms_threshold=0.5,  # More aggressive NMS
    
    max_disappeared=20,  # 0.67 seconds - quick timeout
    min_hits_confirm=4,  # Stricter confirmation
    
    iou_gate=0.35,  # Stricter gate - reduce confusion
    stage1_threshold=0.65,
    
    spatial_gate=0.4,  # Tight spatial constraint
    appearance_gate=0.35,  # Higher appearance threshold
    stage2_threshold=0.55,
    spatial_weight=0.25,
    appearance_weight=0.75,  # Rely more on appearance
    
    feature_ema_alpha=0.92,  # More smoothing for stability
    feature_history_size=40,
    
    reentry_similarity_threshold=0.55,  # Stricter re-entry
    reentry_spatial_gate=150,
    
    exit_best_n_frames=20
)


LOW_FPS_SCENE = SceneConfig(
    """Configuration for 10-15 FPS scenarios"""
    max_people_per_frame=10,
    fps=15,
    resolution=(1920, 1080),
    
    detection_confidence=0.5,
    nms_threshold=0.4,
    
    max_disappeared=15,  # 1 second at 15fps
    min_hits_confirm=2,  # Lower for slower frame rate
    
    iou_gate=0.25,  # Lower - more motion between frames
    stage1_threshold=0.75,
    
    spatial_gate=0.65,  # Higher - expect more movement
    appearance_gate=0.28,
    stage2_threshold=0.65,
    spatial_weight=0.25,  # Rely less on spatial
    appearance_weight=0.75,  # Rely more on appearance
    
    feature_ema_alpha=0.85,  # Less smoothing
    feature_history_size=15,  # Fewer frames available
    
    reentry_similarity_threshold=0.48,
    reentry_spatial_gate=300,  # Wider - more movement
    
    exit_best_n_frames=8
)


HIGH_ANGLE_SCENE = SceneConfig(
    """Configuration for overhead/high-angle cameras"""
    max_people_per_frame=20,
    fps=30,
    resolution=(1920, 1080),
    
    detection_confidence=0.55,
    nms_threshold=0.45,
    
    max_disappeared=25,
    min_hits_confirm=3,
    
    iou_gate=0.35,  # Higher - less occlusion from angle
    stage1_threshold=0.65,
    
    spatial_gate=0.45,
    appearance_gate=0.32,  # Appearance less reliable from top
    stage2_threshold=0.6,
    spatial_weight=0.4,  # Rely more on spatial
    appearance_weight=0.6,
    
    feature_ema_alpha=0.9,
    feature_history_size=30,
    
    reentry_similarity_threshold=0.48,  # Lower - appearance varies
    reentry_spatial_gate=180,
    
    exit_best_n_frames=15
)


def auto_tune_config(
    max_people_per_frame: int,
    fps: int,
    resolution: Tuple[int, int],
    scene_type: str = 'auto'
) -> SceneConfig:
    """
    Automatically select and tune configuration based on scene parameters.
    
    Args:
        max_people_per_frame: Maximum number of people expected
        fps: Camera frame rate
        resolution: (width, height)
        scene_type: 'sparse', 'medium', 'crowded', 'low_fps', 'high_angle', or 'auto'
    
    Returns:
        Tuned SceneConfig
    """
    
    # Auto-detect scene type if not specified
    if scene_type == 'auto':
        if fps < 20:
            scene_type = 'low_fps'
        elif max_people_per_frame <= 5:
            scene_type = 'sparse'
        elif max_people_per_frame <= 20:
            scene_type = 'medium'
        else:
            scene_type = 'crowded'
    
    # Get base config
    base_configs = {
        'sparse': SPARSE_SCENE,
        'medium': MEDIUM_SCENE,
        'crowded': CROWDED_SCENE,
        'low_fps': LOW_FPS_SCENE,
        'high_angle': HIGH_ANGLE_SCENE
    }
    
    config = base_configs.get(scene_type, MEDIUM_SCENE)
    
    # Adjust for actual FPS
    fps_ratio = fps / config.fps
    config.fps = fps
    config.max_disappeared = int(config.max_disappeared * fps_ratio)
    
    # Adjust for resolution
    config.resolution = resolution
    res_ratio = np.sqrt((resolution[0] * resolution[1]) / 
                       (1920 * 1080))
    config.reentry_spatial_gate = int(config.reentry_spatial_gate * res_ratio)
    
    return config


def print_config(config: SceneConfig, name: str = "Current Configuration"):
    """Print configuration in readable format"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    print(f"\n📹 Scene Characteristics:")
    print(f"  Max people per frame: {config.max_people_per_frame}")
    print(f"  FPS: {config.fps}")
    print(f"  Resolution: {config.resolution[0]}x{config.resolution[1]}")
    
    print(f"\n🔍 Detection Parameters:")
    print(f"  Confidence threshold: {config.detection_confidence}")
    print(f"  NMS threshold: {config.nms_threshold}")
    
    print(f"\n🎯 Tracking Parameters:")
    print(f"  Max disappeared: {config.max_disappeared} frames " +
          f"({config.max_disappeared/config.fps:.2f}s)")
    print(f"  Min hits to confirm: {config.min_hits_confirm}")
    
    print(f"\n🔗 Matching - Stage 1 (IoU):")
    print(f"  IoU gate: {config.iou_gate}")
    print(f"  Stage 1 threshold: {config.stage1_threshold}")
    
    print(f"\n🔗 Matching - Stage 2 (Appearance):")
    print(f"  Spatial gate: {config.spatial_gate} (normalized)")
    print(f"  Appearance gate: {config.appearance_gate} (cosine sim)")
    print(f"  Stage 2 threshold: {config.stage2_threshold}")
    print(f"  Weights: {config.spatial_weight:.0%} spatial, " +
          f"{config.appearance_weight:.0%} appearance")
    
    print(f"\n🎨 Feature Parameters:")
    print(f"  EMA alpha: {config.feature_ema_alpha}")
    print(f"  History size: {config.feature_history_size} frames")
    
    print(f"\n🚪 Re-entry Parameters:")
    print(f"  Similarity threshold: {config.reentry_similarity_threshold}")
    print(f"  Spatial gate: {config.reentry_spatial_gate} pixels")
    print(f"  Exit signature frames: {config.exit_best_n_frames}")
    
    print(f"\n{'='*60}\n")


def create_custom_config(
    scene_density: str = 'medium',
    fps: int = 30,
    **overrides
) -> SceneConfig:
    """
    Create custom configuration with overrides.
    
    Args:
        scene_density: 'sparse', 'medium', or 'crowded'
        fps: Camera frame rate
        **overrides: Override specific parameters
    
    Example:
        config = create_custom_config(
            scene_density='crowded',
            fps=25,
            appearance_gate=0.4,
            reentry_similarity_threshold=0.6
        )
    """
    base_configs = {
        'sparse': SPARSE_SCENE,
        'medium': MEDIUM_SCENE,
        'crowded': CROWDED_SCENE
    }
    
    config = base_configs.get(scene_density, MEDIUM_SCENE)
    
    # Update FPS and related parameters
    if fps != config.fps:
        fps_ratio = fps / config.fps
        config.fps = fps
        config.max_disappeared = int(config.max_disappeared * fps_ratio)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown parameter '{key}' ignored")
    
    return config


# GPU-specific recommendations
GPU_CONFIGS = {
    'RTX 4090': {
        'yolo_batch': 1,
        'osnet_batch': 32,
        'use_fp16': True,
        'expected_fps': '60-80 (crowded), 100+ (sparse)'
    },
    'RTX 4080': {
        'yolo_batch': 1,
        'osnet_batch': 24,
        'use_fp16': True,
        'expected_fps': '50-70 (crowded), 90+ (sparse)'
    },
    'RTX 4070': {
        'yolo_batch': 1,
        'osnet_batch': 16,
        'use_fp16': True,
        'expected_fps': '40-60 (crowded), 70+ (sparse)'
    },
    'RTX 3090': {
        'yolo_batch': 1,
        'osnet_batch': 24,
        'use_fp16': True,
        'expected_fps': '45-65 (crowded), 85+ (sparse)'
    },
    'RTX 3080': {
        'yolo_batch': 1,
        'osnet_batch': 20,
        'use_fp16': True,
        'expected_fps': '40-55 (crowded), 75+ (sparse)'
    },
    'RTX 3070': {
        'yolo_batch': 1,
        'osnet_batch': 16,
        'use_fp16': True,
        'expected_fps': '35-50 (crowded), 65+ (sparse)'
    },
    'RTX 3060': {
        'yolo_batch': 1,
        'osnet_batch': 12,
        'use_fp16': True,
        'expected_fps': '30-40 (crowded), 55+ (sparse)'
    },
    'GTX 1660': {
        'yolo_batch': 1,
        'osnet_batch': 8,
        'use_fp16': False,
        'expected_fps': '20-30 (crowded), 40+ (sparse)'
    }
}


def print_gpu_recommendations(gpu_model: str):
    """Print GPU-specific recommendations"""
    if gpu_model in GPU_CONFIGS:
        config = GPU_CONFIGS[gpu_model]
        print(f"\n🎮 Recommendations for {gpu_model}:")
        print(f"  YOLO batch size: {config['yolo_batch']}")
        print(f"  OSNet batch size: {config['osnet_batch']}")
        print(f"  Use FP16: {config['use_fp16']}")
        print(f"  Expected FPS: {config['expected_fps']}")
    else:
        print(f"\n⚠️  No specific config for {gpu_model}")
        print(f"  Available GPUs: {', '.join(GPU_CONFIGS.keys())}")


# Example usage
if __name__ == "__main__":
    print("Scene Configuration Examples:\n")
    
    # Example 1: Auto-tune
    print("\n" + "="*60)
    print("Example 1: Auto-tuned configuration")
    print("="*60)
    config = auto_tune_config(
        max_people_per_frame=12,
        fps=25,
        resolution=(1920, 1080)
    )
    print_config(config, "Auto-tuned Configuration")
    
    # Example 2: Custom config
    print("\n" + "="*60)
    print("Example 2: Custom crowded scene")
    print("="*60)
    config = create_custom_config(
        scene_density='crowded',
        fps=30,
        appearance_gate=0.4,
        reentry_similarity_threshold=0.6
    )
    print_config(config, "Custom Crowded Scene")
    
    # Example 3: GPU recommendations
    print("\n" + "="*60)
    print("Example 3: GPU Recommendations")
    print("="*60)
    print_gpu_recommendations('RTX 4080')
    print_gpu_recommendations('RTX 3070')
