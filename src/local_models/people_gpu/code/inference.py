# inference.py
"""
SageMaker Inference Handler for People Counting System - Version 2
Fixed camera-specific counting and ID reallocation issues
Optimized for ml.g4dn.xlarge instance with enhanced error handling
Model: people8s.pt (fixed model name)
"""

import os
import sys
import json
import time
import base64
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from collections import defaultdict

import numpy as np
import cv2

# Setup logging FIRST
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core dependencies with logging
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch available")
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("Ultralytics available")
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    logger.error("Ultralytics not available")



# Import people counting system with error handling
try:
    # Add current directory to path for local imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from people_counting import PeopleCountingSystemManager
    PEOPLE_COUNTING_AVAILABLE = True
    logger.info("People counting system available")
except ImportError as e:
    PeopleCountingSystemManager = None
    PEOPLE_COUNTING_AVAILABLE = False
    logger.error(f"People counting system not available: {e}")

# Global variables - thread-safe
yolo_model = None
system_manager = None
model_loaded = False
device = None
_lock = threading.Lock()

def find_model_file(model_dir: str) -> str:
    """Find people8s.pt model file with detailed logging"""
    logger.info(f"Searching for people8s.pt in: {model_dir}")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    contents = os.listdir(model_dir)
    logger.info(f"Directory contents: {contents}")
    
    # Fixed model name: people8s.pt
    model_path = os.path.join(model_dir, "people8s.pt")
    
    if os.path.exists(model_path) and os.path.isfile(model_path):
        logger.info(f"Found required model: people8s.pt")
        return model_path
    
    # Fallback: look for any .pt file if people8s.pt not found
    pt_files = [f for f in contents if f.endswith('.pt')]
    if pt_files:
        selected = os.path.join(model_dir, pt_files[0])
        logger.warning(f"people8s.pt not found, using fallback: {pt_files[0]}")
        return selected
    
    raise FileNotFoundError(f"Required model people8s.pt not found in {model_dir}")



def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load model with comprehensive error handling"""
    global yolo_model, system_manager, model_loaded, device
    
    with _lock:
        if model_loaded:
            logger.info("Model already loaded")
            return {"status": "already_loaded"}
        
        logger.info(f"Starting model loading from: {model_dir}")
        
        try:
            # Check critical dependencies
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("ultralytics not available")
            if not PEOPLE_COUNTING_AVAILABLE:
                raise ImportError("people_counting module not available")
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device configured: {device}")
            
            # Find people8s.pt model file
            model_path = find_model_file(model_dir)
            file_size = os.path.getsize(model_path) / 1024**2
            logger.info(f"Loading model: {model_path} ({file_size:.1f}MB)")
            
            # Load YOLO model
            yolo_model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            
            # Move to device if GPU available
            if hasattr(yolo_model, 'model') and str(device) != "cpu":
                try:
                    yolo_model.model.to(device)
                    logger.info(f"Model moved to {device}")
                except Exception as e:
                    logger.warning(f"Failed to move to GPU: {e}")
                    device = "cpu"
            
            # Test model
            try:
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                test_result = yolo_model.predict(dummy_frame, conf=0.5, verbose=False)
                logger.info("Model test successful")
            except Exception as e:
                logger.warning(f"Model test failed: {e}")
            
            # Initialize system manager
            config = {
                "confidence_threshold": 0.35,
                "device": str(device),
                "max_stored_features": 15,
                "similarity_threshold": 0.65
            }
            
            system_manager = PeopleCountingSystemManager(yolo_model, config)
            model_loaded = True
            
            # Prepare response
            gpu_info = {}
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
                }
            
            response = {
                "status": "loaded",
                "device": str(device),
                "model_size_mb": round(file_size, 1),
                "gpu_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
                "model_name": "people8s.pt",
                **gpu_info
            }
            
            logger.info(f"Model loading complete: {response}")
            return response
            
        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            logger.error(error_msg)
            
            # Reset state
            yolo_model = None
            system_manager = None
            model_loaded = False
            device = None
            
            return {
                "status": "failed",
                "error": error_msg,
                "model_dir_exists": os.path.exists(model_dir),
                "model_dir_contents": os.listdir(model_dir) if os.path.exists(model_dir) else []
            }

def input_fn(request_body: str, content_type: str = "application/json") -> Dict[str, Any]:
    """Parse and validate input"""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    
    try:
        data = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    # Required fields
    required = ["camid", "org_id", "userid", "encoding", "threshold", "alert_rate", "return_annotated", "confidence_threshold"]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    
    # Type conversion and validation
    try:
        data["camid"] = int(data["camid"])
        data["org_id"] = int(data["org_id"])
        data["userid"] = int(data["userid"])
        data["threshold"] = int(data["threshold"])
        data["alert_rate"] = float(data["alert_rate"])
        data["return_annotated"] = bool(data["return_annotated"])
        data["confidence_threshold"] = float(data["confidence_threshold"])
        
        # Validate ranges
        if not (0.01 <= data["confidence_threshold"] <= 0.99):
            raise ValueError("confidence_threshold must be between 0.01 and 0.99")
        if not (0.0 <= data["alert_rate"] <= 100.0):
            raise ValueError("alert_rate must be between 0 and 100")
        if data["threshold"] < 1:
            raise ValueError("threshold must be >= 1")
        if not data["encoding"]:
            raise ValueError("encoding cannot be empty")
            
    except (ValueError, TypeError) as e:
        raise ValueError(f"Validation failed: {e}")
    
    return data

def predict_fn(input_data: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Main prediction function with camera-specific processing"""
    start_time = time.time()
    
    # Check initialization
    if not model_loaded or system_manager is None:
        return create_error_response(input_data, "Model not initialized", start_time)
    
    # Extract parameters
    camid = input_data["camid"]
    org_id = input_data["org_id"]
    userid = input_data["userid"]
    threshold = input_data["threshold"]
    alert_rate = input_data["alert_rate"]
    return_annotated = input_data["return_annotated"]
    confidence_threshold = input_data["confidence_threshold"]
    encoding = input_data["encoding"]
    
    try:
        # Decode image
        try:
            image_data = base64.b64decode(encoding)
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image format")
            if frame.shape[0] < 10 or frame.shape[1] < 10:
                raise ValueError("Image too small")
        except Exception as e:
            raise ValueError(f"Image decode failed: {e}")
        
        # Clear GPU cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process frame using camera-specific system manager
        result = system_manager.process_frame(
            frame=frame,
            camid=camid,
            org_id=org_id,
            userid=userid,
            threshold=threshold,
            alert_rate=alert_rate,
            return_annotated=return_annotated,
            confidence_threshold=confidence_threshold
        )
        
        

        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return create_error_response(input_data, str(e), start_time)

def create_error_response(input_data: Dict[str, Any], error_msg: str, start_time: float) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "camid": input_data.get("camid", 0),
        "org_id": input_data.get("org_id", 0),
        "userid": input_data.get("userid", 0),
        "Frame_Id": f"ERROR_{int(time.time() * 1000)}",
        "Time_stamp": datetime.now(timezone.utc).isoformat(),
        "Total_people_detected": 0,
        "Current_occupancy": 0,
        "People_ids": [],
        "Entry_time": [],
        "People_dwell_time": [],
        "Confidence_scores": [],
        "Bounding_boxes": [],
        "x": [], "y": [], "w": [], "h": [],
        "accuracy": [],
        "Total_entries": 0,
        "Total_exits": 0,
        "Net_count": 0,
        "Occupancy_percentage": 0.0,
        "Average_dwell_time": 0.0,
        "Max_occupancy": input_data.get("threshold", 1),
        "Status": "Error",
        "is_alert_triggered": False,
        "Processing_Status": 0,
        "annotated_frame": None,
        "error_message": error_msg[:200],
        "processing_time_ms": round((time.time() - start_time) * 1000, 2)
    }

def output_fn(prediction: Dict[str, Any], content_type: str = "application/json") -> str:
    """Format output as JSON"""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    
    try:
        # Clean data for JSON serialization
        cleaned = {}
        for k, v in prediction.items():
            if v is not None:
                if isinstance(v, (np.integer, np.int32, np.int64)):
                    cleaned[k] = int(v)
                elif isinstance(v, (np.floating, np.float32, np.float64)):
                    cleaned[k] = float(v)
                elif isinstance(v, np.ndarray):
                    cleaned[k] = v.tolist()
                else:
                    cleaned[k] = v
        
        return json.dumps(cleaned, separators=(",", ":"))
        
    except Exception as e:
        logger.error(f"Output formatting failed: {e}")
        return json.dumps({
            "status": "output_error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

if __name__ == "__main__":

    import numpy as np
    from PIL import Image
    import base64

    print("[INFO] Running in local video test mode...")

    # ---------- Setup ----------
    model_dir = r"C:\Users\uct\Desktop\final_sagemaker\people_gpu"  # folder where people8s.pt is located
    model_info = model_fn(model_dir)
    print(f"[INFO] Model load response: {model_info}")

    if model_info.get("status") != "loaded":
        raise RuntimeError(f"Model failed to load: {model_info.get('error', 'Unknown error')}")

    print("[INFO] Model loaded successfully. Starting inference...")

    # ---------- Load video ----------
    video_path = r"C:\Users\uct\Desktop\final_sagemaker\code\istockphoto-1404365178-640_adpp_is.mp4"  # change this to your file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_output = False
    out = None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    if save_output:
        out = cv2.VideoWriter("annotated_video.mp4", fourcc, fps, (frame_width, frame_height))

    print("[INFO] Starting video inference... Press 'q' to quit")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Convert frame to base64 as expected by predict_fn
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Build mock input payload (mimics SageMaker request)
        input_payload = {
            "camid": 1,
            "org_id": 1,
            "userid": 101,
            "encoding": frame_base64,
            "threshold": 10,
            "alert_rate": 0.5,
            "return_annotated": True,
            "confidence_threshold": 0.4
        }

        # Run inference
        output = predict_fn(input_payload, model_info)
        print(output)


        # Decode annotated frame (if returned)
        annotated_frame = frame
        if output.get("annotated_frame"):
            try:
                annotated_bytes = base64.b64decode(output["annotated_frame"])
                nparr = np.frombuffer(annotated_bytes, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    annotated_frame = decoded
            except Exception as e:
                print(f"[WARN] Annotated frame decode failed: {e}")

        # Display
        cv2.imshow("People Counting", annotated_frame)
        if save_output and out:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Inference completed. Processed {frame_count} frames. Saved as annotated_video.mp4")

