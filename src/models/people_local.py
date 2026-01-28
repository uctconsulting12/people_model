import cv2
import json
import base64
import numpy as np
import logging
import os
from dotenv import load_dotenv

import sys, os

# Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.local_models.people_gpu.code.inference import model_fn, predict_fn,output_fn

# ---------- Setup ----------
# model_dir = r"E:\All_models\people_model\src\local_models\people_gpu"  # folder where people8s.pt is located


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "..", "local_models", "people_gpu")
model_dir = os.path.abspath(model_dir)

model_info = model_fn(model_dir)
#------------------------------------------------------------------------------- PPE Detection ------------------------------------------------------------------------------

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)




# -------------------------------------------------------------------------------
#  People counting
# -------------------------------------------------------------------------------





logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)


def people_counting(frame, camera_id, user_id, org_id, threshold, alert_rate):
    """Send a frame to SageMaker endpoint and return (result, error_message, annotated_frame) safely."""
    try:

        # Build mock input payload (mimics SageMaker request)
        input_payload = {
            "camid": camera_id,
            "org_id": org_id,
            "userid": user_id,
            "encoding": frame,
            "threshold": 10,
            "alert_rate": 0.5,
            "return_annotated": True,
            "confidence_threshold": 0.4
        }


        try:
            output = predict_fn(input_payload, model_info)
            return output,None

        except (json.JSONDecodeError, AttributeError) as e:
            msg = f"Invalid JSON response from SageMaker: {e}"
            logger.error(msg)
            return None, msg

    except Exception as e:
        msg = f"Unexpected error in people_counting: {e}"
        logger.exception(msg)
        return None, msg

    except Exception as e:
        msg = f"Unexpected error in ppe_detection: {str(e)}"
        logger.exception(msg)
        return None, msg, None