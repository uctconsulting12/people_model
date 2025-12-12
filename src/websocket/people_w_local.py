import cv2
import json
import time
import asyncio
import logging
import sys
import os
import base64
import requests
from concurrent.futures import ThreadPoolExecutor
# from src.models.people_counting import people_counting

from src.models.people_local import people_counting
from src.store_s3.people_store import upload_to_s3
from src.database.people_query import insert_people_counting

logger = logging.getLogger("people_counting")
logger.setLevel(logging.INFO)


# def  fetch_alert(org_id,user_id):
#       try:
#        response = requests.get(f"https://cctvaidemo.uctconsulting.com/alerts/check-user-define-alert-for-model?org_id={org_id}&user_id={user_id}&model_name=people_counting").content
#        return response

#       except Exception as e:
#           return e
    



def run_peoplecounting_detection(
    client_id: str,
    video_url: str,
    camera_id: int,
    user_id: int,
    org_id: int,
    threshold: int,
    alert_rate: int,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor: ThreadPoolExecutor,
):
    """
    Runs People Counting detection in a separate thread.
    Sends WebSocket messages safely and stores frames to S3/DB in background threads to avoid blocking inference.
    """

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"[{client_id}] Unable to open video stream: {video_url}")
        return

    frame_num = 0

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_num += 1

        # Convert frame to base64 as expected by predict_fn
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        try:
            
            # ---------------- People Counting Inference ----------------
            detections, error = people_counting(frame_base64, camera_id, user_id, org_id, threshold, alert_rate)
            
            ws = sessions.get(client_id, {}).get("ws")

            if detections:
                payload = {"detections": detections}

                # ---------------- WebSocket Send ----------------
                if ws:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps(payload)),
                            loop
                        )
                        logger.info(f"[{client_id}] Frame {frame_num}: Detections sent to client")
                    except Exception as e:
                        logger.error(f"[{client_id}] Frame {frame_num}: WebSocket send error -> {e}")
                        break

                #---------------- Background Storage ----------------
                annotated_frame = detections.get("annotated_frame")
                if annotated_frame is not None and frame_num % 20 == 0:
                    def store_frame():
                        try:
                            s3_url = upload_to_s3(annotated_frame, frame_num)
                            insert_people_counting(detections, s3_url)
                            logger.info(f"[{client_id}] Frame {frame_num} stored successfully")
                        except Exception as e:
                            logger.error(f"[{client_id}] Frame {frame_num} store error -> {e}")

                    # Schedule S3/DB storage without blocking inference
                    storage_executor.submit(store_frame)

            else:
                if ws:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps({"success": False, "message": error})),
                        loop
                    )
                logger.warning(f"[{client_id}] Frame {frame_num}: No detections - {error}")
                break

        except Exception as e:
            logger.exception(f"[{client_id}] Frame {frame_num}: Pipeline error -> {e}")

    cap.release()
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] People counting stopped and resources released")






# if __name__ == "__main__":
#     print(fetch_alert(4,4))