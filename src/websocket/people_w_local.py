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
from multiprocessing import Process, Queue

from src.models.people_local import people_counting
from src.store_s3.people_store import upload_to_s3
from src.database.people_query import insert_people_counting

logger = logging.getLogger("people_counting")
logger.setLevel(logging.INFO)


def run_storage_worker(q, client_id):
    """
    Runs in a SEPARATE PROCESS.
    Handles S3 upload + DB insert.
    """

    logger.info(f"[{client_id}] Storage worker started.")

    while True:
        item = q.get()

        # Sentinel: exit
        if item is None:
            break

        frame_id, annotated_frame, detections = item

        try:
            # Upload to S3
            s3_url = upload_to_s3(annotated_frame, frame_id)

            # DB insert
            insert_people_counting(detections, s3_url)

            logger.info(f"[{client_id}] Stored frame {frame_id}")

        except Exception as e:
            logger.error(f"[{client_id}] Error storing frame {frame_id}: {e}")

    logger.info(f"[{client_id}] Storage worker exiting...")
    



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
     # ---------------------------------------------------------
    # START MULTIPROCESS STORAGE WORKER
    # ---------------------------------------------------------
    store_queue = Queue(maxsize=1000)

    storage_process = Process(
        target=run_storage_worker,
        args=(store_queue, client_id),
        daemon=True
    )
    storage_process.start()

    logger.info(f"[{client_id}] Storage worker process started.")

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
                    if annotated_frame is not None:
                        # JSON COPY to avoid race condition
                        safe_copy = json.loads(json.dumps(detections))

                        try:
                            store_queue.put_nowait(
                                (frame_num, annotated_frame, safe_copy)
                            )
                        except:
                            logger.warning(
                                f"[{client_id}] Storage queue full; frame {frame_num} dropped."
                            )

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
    # STOP STORAGE PROCESS
    store_queue.put(None)
    storage_process.join(timeout=5)
    
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] People counting stopped and resources released")






# if __name__ == "__main__":
#     print(fetch_alert(4,4))