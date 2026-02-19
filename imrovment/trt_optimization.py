"""
TensorRT Optimization Guide for People Counting Pipeline

This module provides:
1. TensorRT conversion utilities for YOLO and OSNet
2. Batched inference for OSNet features
3. Performance benchmarking tools
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import time


class TensorRTConfig:
    """Configuration for TensorRT optimization"""
    # YOLO settings
    YOLO_INPUT_SIZE = (640, 640)  # Fixed size for TensorRT
    YOLO_FP16 = True
    YOLO_BATCH_SIZE = 1
    
    # OSNet settings
    OSNET_INPUT_SIZE = (256, 128)  # Width x Height
    OSNET_FP16 = True
    OSNET_MAX_BATCH = 32  # Batch multiple detections
    
    # Optimization flags
    USE_CUDA_GRAPHS = True  # For static models
    WARMUP_ITERATIONS = 10


class YOLODetectorTRT:
    """
    TensorRT-optimized YOLO detector.
    
    To convert YOLO to TensorRT:
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.export(format='engine', imgsz=640, half=True)
    # This creates yolov8n.engine
    ```
    """
    def __init__(self, engine_path: str, conf_threshold: float = 0.5):
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        
        # Load engine (example using Ultralytics)
        from ultralytics import YOLO
        self.model = YOLO(engine_path, task='detect')
        
        self.input_size = TensorRTConfig.YOLO_INPUT_SIZE
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup inference"""
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(TensorRTConfig.WARMUP_ITERATIONS):
            self.detect(dummy_img)
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect people in frame.
        Returns list of detections with bbox and confidence.
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for person class (class 0 in COCO)
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf
                    })
        
        return detections


class OSNetReIDTRT:
    """
    TensorRT-optimized OSNet ReID with batched inference.
    
    To convert OSNet to TensorRT:
    1. Export to ONNX:
       ```python
       import torch
       model = osnet_x1_0(pretrained=True)
       model.eval()
       dummy_input = torch.randn(1, 3, 256, 128).cuda()
       torch.onnx.export(
           model, dummy_input, 'osnet.onnx',
           input_names=['input'], output_names=['output'],
           dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
       )
       ```
    
    2. Build TensorRT engine:
       ```bash
       trtexec --onnx=osnet.onnx --saveEngine=osnet_fp16.engine \
               --fp16 --minShapes=input:1x3x256x128 \
               --optShapes=input:8x3x256x128 \
               --maxShapes=input:32x3x256x128
       ```
    """
    def __init__(self, engine_path: Optional[str] = None, use_torch: bool = True):
        self.engine_path = engine_path
        self.use_torch = use_torch  # Fallback to PyTorch if no TRT
        self.input_size = TensorRTConfig.OSNET_INPUT_SIZE
        self.max_batch = TensorRTConfig.OSNET_MAX_BATCH
        
        if engine_path and Path(engine_path).exists():
            self._load_trt_engine()
        else:
            self._load_torch_model()
        
        # Warmup
        self._warmup()
    
    def _load_trt_engine(self):
        """Load TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.logger = trt.Logger(trt.Logger.WARNING)
            
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
            
            self.use_trt = True
            print(f"Loaded TensorRT engine from {self.engine_path}")
            
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            print("Falling back to PyTorch")
            self._load_torch_model()
    
    def _load_torch_model(self):
        """Load PyTorch OSNet model"""
        from torchreid.models import build_model
        
        self.model = build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if TensorRTConfig.OSNET_FP16:
                self.model = self.model.half()
        
        self.use_trt = False
        print("Loaded PyTorch OSNet model")
    
    def _warmup(self):
        """Warmup inference"""
        dummy_crops = [np.zeros((128, 256, 3), dtype=np.uint8) 
                      for _ in range(4)]
        for _ in range(TensorRTConfig.WARMUP_ITERATIONS):
            self.extract_features_batch(dummy_crops)
    
    def extract_features_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features for batch of crops.
        This is the key optimization - batch all detections per frame.
        
        Args:
            crops: List of person crops (H, W, C) in BGR
        
        Returns:
            features: (N, feature_dim) numpy array
        """
        if len(crops) == 0:
            return np.array([])
        
        # Preprocess all crops
        processed = self._preprocess_batch(crops)
        
        # Split into batches if needed
        all_features = []
        for i in range(0, len(processed), self.max_batch):
            batch = processed[i:i + self.max_batch]
            
            if self.use_trt:
                features = self._inference_trt(batch)
            else:
                features = self._inference_torch(batch)
            
            all_features.append(features)
        
        return np.vstack(all_features)
    
    def _preprocess_batch(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess batch of crops"""
        processed = []
        
        for crop in crops:
            # Resize
            img = cv2.resize(crop, self.input_size)
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / \
                  np.array([0.229, 0.224, 0.225])
            # HWC to CHW
            img = np.transpose(img, (2, 0, 1))
            processed.append(img)
        
        # Stack to batch
        batch = torch.from_numpy(np.stack(processed)).float()
        
        if torch.cuda.is_available():
            batch = batch.cuda()
            if TensorRTConfig.OSNET_FP16 and not self.use_trt:
                batch = batch.half()
        
        return batch
    
    def _inference_torch(self, batch: torch.Tensor) -> np.ndarray:
        """PyTorch inference"""
        with torch.no_grad():
            features = self.model(batch)
            features = features.cpu().numpy()
        
        # L2 normalize
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        return features
    
    def _inference_trt(self, batch: torch.Tensor) -> np.ndarray:
        """TensorRT inference"""
        # Placeholder - implement based on your TRT wrapper
        # This would use the TensorRT engine for inference
        raise NotImplementedError("TensorRT inference not implemented")


class PerformanceBenchmark:
    """Benchmark tool for pipeline optimization"""
    
    def __init__(self):
        self.metrics = {
            'detection': [],
            'reid_extraction': [],
            'tracking': [],
            'total': []
        }
    
    def benchmark_detector(self, detector, test_images: List[np.ndarray], 
                          iterations: int = 100):
        """Benchmark YOLO detector"""
        print(f"\n=== Benchmarking Detector ===")
        
        times = []
        for img in test_images[:iterations]:
            start = time.time()
            detections = detector.detect(img)
            times.append(time.time() - start)
        
        self._print_stats("Detection", times)
        return times
    
    def benchmark_reid(self, reid_model, test_crops: List[np.ndarray],
                      batch_sizes: List[int] = [1, 4, 8, 16, 32]):
        """Benchmark ReID feature extraction with different batch sizes"""
        print(f"\n=== Benchmarking ReID ===")
        
        for batch_size in batch_sizes:
            times = []
            for i in range(0, len(test_crops), batch_size):
                batch = test_crops[i:i+batch_size]
                if len(batch) == 0:
                    continue
                
                start = time.time()
                features = reid_model.extract_features_batch(batch)
                times.append(time.time() - start)
            
            self._print_stats(f"ReID (batch={batch_size})", times)
    
    def benchmark_full_pipeline(self, counter, test_frames: List[np.ndarray]):
        """Benchmark full pipeline"""
        print(f"\n=== Benchmarking Full Pipeline ===")
        
        times = []
        for frame in test_frames:
            start = time.time()
            results = counter.process_frame(frame)
            times.append(time.time() - start)
        
        self._print_stats("Full Pipeline", times)
        
        # Print FPS statistics
        fps_values = [1.0 / t for t in times if t > 0]
        print(f"\nFPS Statistics:")
        print(f"  Mean FPS: {np.mean(fps_values):.1f}")
        print(f"  Min FPS: {np.min(fps_values):.1f}")
        print(f"  Max FPS: {np.max(fps_values):.1f}")
        print(f"  Std FPS: {np.std(fps_values):.1f}")
    
    def _print_stats(self, name: str, times: List[float]):
        """Print timing statistics"""
        times = np.array(times) * 1000  # Convert to ms
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(times):.2f} ms")
        print(f"  Median: {np.median(times):.2f} ms")
        print(f"  Std: {np.std(times):.2f} ms")
        print(f"  Min: {np.min(times):.2f} ms")
        print(f"  Max: {np.max(times):.2f} ms")
        print(f"  P95: {np.percentile(times, 95):.2f} ms")
        print(f"  P99: {np.percentile(times, 99):.2f} ms")


def convert_yolo_to_trt(model_path: str, output_path: str):
    """
    Helper function to convert YOLO to TensorRT.
    
    Usage:
        convert_yolo_to_trt('yolov8n.pt', 'yolov8n.engine')
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    model.export(
        format='engine',
        imgsz=640,
        half=True,
        simplify=True,
        workspace=4  # GB
    )
    print(f"TensorRT engine saved to {output_path}")


def convert_osnet_to_onnx(model, output_path: str):
    """
    Helper function to convert OSNet to ONNX.
    
    Usage:
        from torchreid.models import build_model
        model = build_model('osnet_x1_0', num_classes=1000, pretrained=True)
        convert_osnet_to_onnx(model, 'osnet.onnx')
    """
    import torch
    
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 128)
    
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=13
    )
    print(f"ONNX model saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("""
    TensorRT Optimization Steps:
    
    1. Convert YOLO to TensorRT:
       python -c "from trt_optimization import convert_yolo_to_trt; \\
                  convert_yolo_to_trt('yolov8n.pt', 'yolov8n.engine')"
    
    2. Convert OSNet to ONNX:
       python -c "from torchreid.models import build_model; \\
                  from trt_optimization import convert_osnet_to_onnx; \\
                  model = build_model('osnet_x1_0', num_classes=1000, pretrained=True); \\
                  convert_osnet_to_onnx(model, 'osnet.onnx')"
    
    3. Build OSNet TensorRT engine:
       trtexec --onnx=osnet.onnx --saveEngine=osnet_fp16.engine \\
               --fp16 --minShapes=input:1x3x256x128 \\
               --optShapes=input:8x3x256x128 \\
               --maxShapes=input:32x3x256x128
    
    4. Use optimized models:
       detector = YOLODetectorTRT('yolov8n.engine')
       reid_model = OSNetReIDTRT('osnet_fp16.engine')
    """)
