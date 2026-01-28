# people_counting.py
"""

trying ti revert old

================================================================================
PEOPLE COUNTING SYSTEM - PRODUCTION VERSION
================================================================================

OVERVIEW:
---------
This module implements a sophisticated people counting system with:
- YOLO-based person detection
- Deep learning re-identification (OSNet)
- Multi-object tracking (DeepSORT)
- State-based alert debouncing
- Dual STATUS/ALERT system

SIMPLIFIED DUAL STATUS/ALERT SYSTEM:
------------------------------------
1. STATUS Field (Backward Compatible):
   - "High Occupancy": occupancy >= critical_threshold (user-defined)
   - "Medium Occupancy": occupancy >= 65% (FIXED)
   - "Low Occupancy": occupancy < 65%

2. ALERT Field (New with Intelligent Debouncing):
   - "Critical": Shown ONCE per incident when occupancy >= user's alert_rate
   - "Warning": Shown EVERY frame when 65% <= occupancy < alert_rate
   - "": No alert or suppressed critical

3. is_alert_triggered:
   - True: ONLY when a NEW critical alert is sent
   - False: For suppressed critical, warnings, or normal operation

THRESHOLD CALCULATIONS:
-----------------------
critical_people = round(threshold * (alert_rate / 100))  # User controlled
warning_people = round(threshold * 0.65)                 # FIXED at 65%
clear_people = warning_people                            # Same as warning

EXAMPLES:
---------
Shop A (capacity=20, alert_rate=90):
  - Critical at: round(20 * 0.90) = 18 people
  - Warning at: round(20 * 0.65) = 13 people
  - Clear at: 13 people

Shop B (capacity=100, alert_rate=80):
  - Critical at: round(100 * 0.80) = 80 people
  - Warning at: round(100 * 0.65) = 65 people
  - Clear at: 65 people

Shop C (capacity=600, alert_rate=70):
  - Critical at: round(600 * 0.70) = 420 people
  - Warning at: round(600 * 0.65) = 390 people
  - Clear at: 390 people

CRITICAL ALERT STATE MACHINE:
-----------------------------
INITIAL → CRITICAL ACTIVE (occupancy >= critical, first time)
CRITICAL ACTIVE → CRITICAL ACTIVE (occupancy >= critical, suppressed)
CRITICAL ACTIVE → INITIAL (occupancy < 65%, cleared)

Author: AI-Powered People Counting Team
Version: 1.0 - Production
Date: 2024-12-17
================================================================================
"""

import time
import base64
import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque

import numpy as np
import cv2

# Import ReID components
from osnet_deepsort_reid import ImprovedReIdentifier, RobustTracker, Track

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timestamp() -> str:
    """
    Get current UTC timestamp in ISO 8601 format.

    Returns:
        str: ISO formatted timestamp (e.g., "2024-12-17T10:30:45.123456Z")
    """
    return datetime.now(timezone.utc).isoformat()


def unix_to_iso(unix_timestamp: float) -> str:
    """
    Convert Unix timestamp to ISO 8601 format string.

    Args:
        unix_timestamp (float): Unix timestamp (seconds since epoch)

    Returns:
        str: ISO formatted timestamp or current time if conversion fails

    Example:
        >>> unix_to_iso(1702817445.123)
        "2024-12-17T10:30:45.123000Z"
    """
    try:
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc).isoformat()
    except Exception as e:
        logger.debug(f"Failed to convert timestamp {unix_timestamp}: {e}")
        return datetime.now(timezone.utc).isoformat()


def seconds_to_hhmmss(seconds):
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds (float or int): Duration in seconds

    Returns:
        str: Time in HH:MM:SS format (e.g., "00:05:30")

    Examples:
        >>> seconds_to_hhmmss(5)
        "00:00:05"
        >>> seconds_to_hhmmss(90)
        "00:01:30"
        >>> seconds_to_hhmmss(3725)
        "01:02:05"
        >>> seconds_to_hhmmss(86400)
        "23:59:59"  # Capped at maximum
        >>> seconds_to_hhmmss(-10)
        "00:00:00"  # Negative values become zero

    Notes:
        - Negative values are clamped to 0
        - Values > 23:59:59 (86399 seconds) are capped at 23:59:59
        - Always returns zero-padded format (02d)
    """
    # Clamp negative values to zero
    if seconds < 0:
        seconds = 0

    # Cap at 23:59:59 (86399 seconds)
    if seconds > 86399:
        seconds = 86399

    # Convert to integer seconds
    total_seconds = int(seconds)

    # Calculate hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    # Return formatted string with zero-padding
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# CAMERA PEOPLE COUNTING SYSTEM
# ============================================================================

class CameraPeopleCountingSystem:
    """
    Camera-specific people counting system with SIMPLIFIED DUAL STATUS/ALERT.

    This class manages all counting operations for a single camera, including:
    - Person detection using YOLO
    - Re-identification using OSNet
    - Tracking using DeepSORT
    - Entry/exit tracking
    - Alert state management

    Attributes:
        camera_id (int): Unique camera identifier
        yolo_model: Pre-loaded YOLO model for detection
        tracker (RobustTracker): Multi-object tracker
        reidentifier (ImprovedReIdentifier): Person re-identification system
        frame_count (int): Number of frames processed
        total_entries (int): Total people entered (lifetime)
        total_exits (int): Total people exited (lifetime)
        active_people (dict): Currently present people
        critical_alert_state (dict): Critical alert state for debouncing
    """

    def __init__(self, camera_id: int, yolo_model, config: Dict[str, Any]):
        """
        Initialize camera-specific counting system.

        Args:
            camera_id (int): Unique identifier for this camera
            yolo_model: Pre-loaded YOLO model instance
            config (Dict[str, Any]): Configuration dictionary with:
                - confidence_threshold (float): YOLO confidence (default: 0.35)
                - max_stored_features (int): ReID feature storage (default: 30)
                - similarity_threshold (float): ReID similarity (default: 0.55)
                - osnet_weights_path (str): Path to OSNet weights
        """
        self.camera_id = camera_id
        self.yolo_model = yolo_model
        self.config = config or {}

        # YOLO detection threshold
        self.confidence_threshold = self.config.get("confidence_threshold", 0.35)

        # Initialize ReID components
        # RobustTracker: Handles object tracking across frames
        self.tracker = RobustTracker(max_disappeared=30, max_distance=100.0)

        # ImprovedReIdentifier: Identifies if a person has been seen before
        self.reidentifier = ImprovedReIdentifier(
            max_stored=self.config.get("max_stored_features", 30),
            similarity_threshold=self.config.get("similarity_threshold", 0.55),
            weights_path=self.config.get("osnet_weights_path")
        )

        # State tracking
        self.frame_count = 0  # Total frames processed
        self.total_entries = 0  # Lifetime entry count
        self.total_exits = 0  # Lifetime exit count
        self.active_people = {}  # Currently present people {track_id: person_info}

        # Event logs (using deque for efficient append/pop)
        self.entry_log = deque(maxlen=200)  # Recent entry events
        self.exit_log = deque(maxlen=200)  # Recent exit events
        self.recent_exits = deque(maxlen=100)  # Last 100 exits for API response

        # ============= SIMPLIFIED DUAL STATUS/ALERT SYSTEM =============
        # Warning at FIXED 65%, Critical user-controlled
        # Critical alert state tracking (for debouncing)
        self.critical_alert_state = {
            "critical_alert_active": False,  # Is critical alert currently active?
            "first_triggered_frame": None,  # Frame when critical alert first triggered
        }

        # Top 10 dwell time tracking for 60-minute average
        self.top_10_dwell = deque(maxlen=10)  # Stores top 10 dwell times from last 60 minutes
        self.dwell_60_time = 0.0  # Mean of top_10_dwell (used when no people present)

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"Camera {camera_id} counting system initialized - "
                    f"confidence: {self.confidence_threshold}, "
                    f"reid_threshold: {self.reidentifier.similarity_threshold}, "
                    f"simplified_dual_status_alert: ENABLED (warning fixed at 65%)")

    def set_confidence_threshold(self, threshold: float):
        """
        Update YOLO confidence threshold dynamically.

        Args:
            threshold (float): New confidence threshold (0.01 to 0.99)

        Note:
            Threshold is clamped to valid range [0.01, 0.99]
        """
        self.confidence_threshold = max(0.01, min(0.99, float(threshold)))
        logger.info(f"Camera {self.camera_id}: Updated confidence threshold to {self.confidence_threshold}")

    def detect_people(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """
        Detect people in a frame using YOLO.

        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV)

        Returns:
            Tuple[List[List[float]], List[float]]:
                - boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
                - confidences: List of confidence scores [0.85, 0.92, ...]

        Example:
            >>> boxes, confs = self.detect_people(frame)
            >>> print(f"Detected {len(boxes)} people")
            Detected 5 people
        """
        if frame is None or frame.size == 0:
            logger.warning(f"Camera {self.camera_id}: Invalid frame input")
            return [], []

        try:
            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"Camera {self.camera_id}: Running YOLO with conf={self.confidence_threshold}")

            # Run YOLO detection
            results = self.yolo_model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device='cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
            )

            boxes = []
            confidences = []

            # Extract person detections (class 0)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Person class in COCO dataset
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            boxes.append([float(x1), float(y1), float(x2), float(y2)])
                            confidences.append(conf)

            logger.debug(f"Camera {self.camera_id}: Detected {len(boxes)} people")
            return boxes, confidences

        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Detection failed: {e}")
            return [], []

    def process_frame(self, frame: np.ndarray, threshold: int, alert_rate: int,
                      return_annotated: bool = False) -> Dict[str, Any]:
        """
        Process a single frame with SIMPLIFIED DUAL STATUS/ALERT system.

        This is the main entry point for frame processing. It:
        1. Detects people using YOLO
        2. Extracts ReID features
        3. Updates tracker
        4. Handles entries/exits
        5. Calculates occupancy and alerts
        6. Returns comprehensive metrics

        Args:
            frame (np.ndarray): Input frame (BGR format)
            threshold (int): Maximum capacity (e.g., 100 people)
            alert_rate (int): Critical alert percentage (e.g., 80%)
            return_annotated (bool): Whether to return annotated frame

        Returns:
            Dict[str, Any]: Response dictionary with 27+ fields including:
                - camid, Frame_Id, Time_stamp, Frame_Count
                - Total_people_detected, Current_occupancy
                - People_ids, Entry_time, Exit_time
                - Occupancy_percentage, Over_capacity_count
                - Status, alert, is_alert_triggered
                - Processing_Status, annotated_frame

        THRESHOLD CALCULATIONS:
            critical_people = round(threshold * alert_rate / 100)
            warning_people = round(threshold * 0.65)  # FIXED 65%

        EXAMPLES:
            Shop A: threshold=20, alert_rate=90
              - Critical at: 18 people
              - Warning at: 13 people

            Shop B: threshold=100, alert_rate=80
              - Critical at: 80 people
              - Warning at: 65 people
        """
        with self._lock:
            self.frame_count += 1
            current_time = time.time()
            timestamp = get_timestamp()
            frame_id = f"FR_{self.camera_id}_{int(current_time * 1000)}"

            logger.info(f"Camera {self.camera_id}: Processing frame {self.frame_count}")

            try:
                height, width = frame.shape[:2]

                # Step 1: Detect people using YOLO
                boxes, confidences = self.detect_people(frame)
                logger.info(f"Camera {self.camera_id}: Detected {len(boxes)} people")

                # Step 2: Extract ReID features for each detection
                features_list = []
                for box in boxes:
                    features = self.reidentifier.extract_features(frame, box)
                    features_list.append(features)

                # Step 3: Update tracker with detections and features
                tracks = self.tracker.update(boxes, confidences, features_list)
                logger.info(f"Camera {self.camera_id}: Active tracks: {len(tracks)}")

                # Step 4: Process tracks and handle entries/exits
                people = self._process_tracks(tracks, current_time)
                logger.info(f"Camera {self.camera_id}: Final people count: {len(people)}")

                # Step 5: Calculate metrics with SIMPLIFIED DUAL STATUS/ALERT logic
                metrics = self._calculate_simplified_dual_status_alert(
                    people, threshold, alert_rate
                )

                # Step 6: Extract coordinates from TRACKED PEOPLE (FIXED)
                coords = self._extract_coordinates(people, height, width)
                logger.debug(f"Camera {self.camera_id}: Extracted coordinates from {len(people)} tracked people")

                # Step 7: Generate annotated frame if requested
                annotated_frame_b64 = None
                if return_annotated:
                    try:
                        annotated = self._annotate_frame(frame, people, threshold,
                                                         metrics["occupancy_percentage"],
                                                         metrics["status"],
                                                         metrics["avg_dwell_time"])
                        annotated_frame_b64 = self._frame_to_base64(annotated)
                    except Exception as e:
                        logger.debug(f"Annotation failed for camera {self.camera_id}: {e}")

                # Step 8: Build comprehensive response
                response = {
                    "camid": self.camera_id,
                    "Frame_Id": frame_id,
                    "Time_stamp": timestamp,
                    "Frame_Count": self.frame_count,
                    "Total_people_detected": len(people),
                    "Current_occupancy": metrics["current_occupancy"],
                    "People_ids": [p["id"] for p in people],
                    "Entry_time": [p.get("entry_time_iso", "") for p in people],
                    "Exit_time": self._get_exit_times(),
                    "exitid": self._get_exit_ids(),
                    "People_dwell_time": [p.get("dwell_time_hhmmss", "00:00:00") for p in people],
                    # NEW: HH:MM:SS array
                    "Confidence_scores": [p.get("confidence", 0.0) for p in people],
                    "Bounding_boxes": [p["bbox"] for p in people],
                    "x": coords["x"],
                    "y": coords["y"],
                    "w": coords["w"],
                    "h": coords["h"],
                    "accuracy": [round(p.get("confidence", 0.0), 3) for p in people],
                    "Total_entries": self.total_entries,
                    "Total_exits": self.total_exits,
                    "Net_count": metrics["net_count"],
                    "Occupancy_percentage": metrics["occupancy_percentage"],
                    "Over_capacity_count": metrics["over_capacity_count"],
                    "Average_dwell_time": seconds_to_hhmmss(metrics["avg_dwell_time"]),  # NEW: HH:MM:SS string
                    "Max_occupancy": threshold,
                    "Status": metrics["status"],  # "High Occupancy" or ""
                    "is_alert_triggered": metrics["is_alert_triggered"],  # True for new alerts, False otherwise
                    "Processing_Status": 1,
                    "annotated_frame": annotated_frame_b64
                }

                # Validate data consistency (CRITICAL CHECKS)
                try:
                    assert len(response["People_ids"]) == len(response["x"]), \
                        f"Mismatch: People_ids={len(response['People_ids'])}, x={len(response['x'])}"
                    assert len(response["People_ids"]) == len(response["y"]), \
                        f"Mismatch: People_ids={len(response['People_ids'])}, y={len(response['y'])}"
                    assert len(response["People_ids"]) == len(response["w"]), \
                        f"Mismatch: People_ids={len(response['People_ids'])}, w={len(response['w'])}"
                    assert len(response["People_ids"]) == len(response["h"]), \
                        f"Mismatch: People_ids={len(response['People_ids'])}, h={len(response['h'])}"
                    assert len(response["People_ids"]) == len(response["Bounding_boxes"]), \
                        f"Mismatch: People_ids={len(response['People_ids'])}, Bounding_boxes={len(response['Bounding_boxes'])}"
                    assert len(response["People_ids"]) == len(response["Confidence_scores"]), \
                        f"Mismatch: People_ids={len(response['People_ids'])}, Confidence_scores={len(response['Confidence_scores'])}"

                    # Validate alert logic consistency
                    if response["Current_occupancy"] == 0:
                        assert response["Status"] == "", \
                            f"Status should be empty with 0 occupancy, got '{response['Status']}'"
                        assert response["is_alert_triggered"] == False, \
                            f"is_alert_triggered should be False with 0 occupancy, got {response['is_alert_triggered']}"

                    logger.debug(f"✅ Camera {self.camera_id}: Data consistency validation passed")

                except AssertionError as e:
                    logger.error(f"❌ Camera {self.camera_id}: DATA CONSISTENCY ERROR - {e}")
                    logger.error(f"   People count: {len(people)}")
                    logger.error(f"   Response arrays: People_ids={len(response['People_ids'])}, "
                                 f"x={len(response['x'])}, y={len(response['y'])}, "
                                 f"w={len(response['w'])}, h={len(response['h'])}")
                    # Continue execution but log the error

                logger.info(f"Camera {self.camera_id} Frame {self.frame_count}: {len(people)} people, "
                            f"occupancy: {metrics['occupancy_percentage']:.1f}%, "
                            f"Status: '{metrics['status']}', is_alert_triggered: {metrics['is_alert_triggered']}")

                return response

            except Exception as e:
                logger.error(f"Frame processing failed for camera {self.camera_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Return error response with same structure
                return {
                    "camid": self.camera_id,
                    "Frame_Id": frame_id,
                    "Time_stamp": timestamp,
                    "Frame_Count": self.frame_count,
                    "Total_people_detected": 0,
                    "Current_occupancy": 0,
                    "People_ids": [],
                    "Entry_time": [],
                    "Exit_time": [],
                    "exitid": [],
                    "People_dwell_time": [],
                    "Confidence_scores": [],
                    "Bounding_boxes": [],
                    "x": [], "y": [], "w": [], "h": [],
                    "accuracy": [],
                    "Total_entries": self.total_entries,
                    "Total_exits": self.total_exits,
                    "Net_count": max(0, self.total_entries - self.total_exits),
                    "Occupancy_percentage": 0.0,
                    "Over_capacity_count": 0,
                    "Average_dwell_time": "00:00:00",  # NEW: HH:MM:SS string
                    "Max_occupancy": threshold,
                    "Status": "Error",
                    "is_alert_triggered": False,
                    "Processing_Status": 0,
                    "error_message": str(e)[:200]
                }

    def _process_tracks(self, tracks: Dict[int, Track], current_time: float) -> List[Dict[str, Any]]:
        """
        Process tracking results and handle entry/exit events with re-identification.

        This method:
        1. Detects new tracks (potential entries)
        2. Tries to re-identify people who left and returned
        3. Updates existing tracks
        4. Detects tracks that disappeared (exits)
        5. Records entry/exit events

        Args:
            tracks (Dict[int, Track]): Current active tracks {track_id: Track}
            current_time (float): Unix timestamp of current frame

        Returns:
            List[Dict[str, Any]]: List of people currently in frame, each with:
                - id: Person ID (consistent across re-entry)
                - bbox: Bounding box [x1, y1, x2, y2]
                - confidence: Detection confidence
                - entry_time: Unix timestamp of entry
                - entry_time_iso: ISO formatted entry time
                - dwell_time: Seconds since entry

        RE-IDENTIFICATION LOGIC:
            - New track detected → Try to match with recently exited person
            - If match found → Reuse person ID (re-entry)
            - If no match → Assign new person ID (new entry)

        ENTRY/EXIT EVENTS:
            - Entry: total_entries++, add to entry_log
            - Exit: total_exits++, add to exit_log and recent_exits
        """
        people = []
        current_track_ids = set(tracks.keys())
        previous_track_ids = set(self.active_people.keys())

        # Handle new entries (tracks that appeared this frame)
        new_track_ids = current_track_ids - previous_track_ids
        for track_id in new_track_ids:
            track = tracks[track_id]
            features = track.features
            center = track.center

            # Try to re-identify: Is this someone who was here before?
            matched_id = self.reidentifier.find_match(features, current_time, center)

            if matched_id:
                # Re-identified! This person was here before and left
                person_id = matched_id
                logger.info(f"Camera {self.camera_id}: Re-identified person {person_id}")
            else:
                # New person - assign new ID with camera prefix and count as entry
                self.total_entries += 1
                person_id = f"P{self.camera_id}_{self.total_entries}"
                entry_record = {
                    "person_id": person_id,
                    "entry_timestamp": unix_to_iso(current_time),
                    "entry_time_unix": current_time
                }
                self.entry_log.append(entry_record)
                logger.info(f"Camera {self.camera_id}: New person {person_id} entered "
                            f"(Total entries: {self.total_entries})")

            # Track this person
            self.active_people[track_id] = {
                "id": person_id,
                "entry_time": current_time,
                "last_seen": current_time,
                "features": features,  # Store features for exit re-ID
                "last_center": center  # Store location
            }

        # Update existing tracks (people still present)
        for track_id in current_track_ids & previous_track_ids:
            track = tracks[track_id]
            self.active_people[track_id]["last_seen"] = current_time
            self.active_people[track_id]["features"] = track.features  # Update features
            self.active_people[track_id]["last_center"] = track.center  # Update location

        # Handle exits (tracks that disappeared this frame)
        exited_track_ids = previous_track_ids - current_track_ids
        for track_id in exited_track_ids:
            person_info = self.active_people[track_id]
            person_id = person_info["id"]
            entry_time = person_info["entry_time"]
            dwell_time = current_time - entry_time
            features = person_info.get("features")
            exit_location = person_info.get("last_center", (0, 0))

            # Store features for re-identification when person exits
            if features is not None:
                self.reidentifier.store_features(person_id, features, current_time, exit_location)
                logger.debug(f"Camera {self.camera_id}: Stored features for person {person_id} at exit")

            # Record exit event
            self.total_exits += 1
            exit_record = {
                "person_id": person_id,
                "exit_timestamp": unix_to_iso(current_time),
                "exit_time_unix": current_time,
                "dwell_time_seconds": dwell_time
            }
            self.exit_log.append(exit_record)
            self.recent_exits.append(exit_record)

            # Add dwell time to top_10_dwell list (keeps last 10 automatically via maxlen)
            self.top_10_dwell.append(dwell_time)

            # Update dwell_60_time (mean of top 10)
            if len(self.top_10_dwell) > 0:
                self.dwell_60_time = sum(self.top_10_dwell) / len(self.top_10_dwell)
                logger.info(f"Camera {self.camera_id}: ✅ UPDATED dwell_60_time = {self.dwell_60_time:.2f} "
                            f"(top_10_dwell has {len(self.top_10_dwell)} items: {list(self.top_10_dwell)})")

            logger.info(f"Camera {self.camera_id}: Person {person_id} exited after "
                        f"{dwell_time:.1f}s (Total exits: {self.total_exits})")

            # Remove from active tracking
            del self.active_people[track_id]

        # Build list of currently present people
        for track_id, person_info in self.active_people.items():
            if track_id in tracks:
                track = tracks[track_id]
                dwell_time_seconds = current_time - person_info["entry_time"]
                people.append({
                    "id": person_info["id"],
                    "bbox": [track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3]],
                    "confidence": track.confidence,
                    "entry_time": person_info["entry_time"],
                    "entry_time_iso": unix_to_iso(person_info["entry_time"]),
                    "dwell_time": dwell_time_seconds,  # Keep as float for calculations
                    "dwell_time_hhmmss": seconds_to_hhmmss(dwell_time_seconds)  # NEW: HH:MM:SS string
                })

        return people

    def _calculate_simplified_dual_status_alert(
            self,
            people: List[Dict[str, Any]],
            threshold: int,
            alert_rate: int
    ) -> Dict[str, Any]:
        """
        Calculate metrics with SIMPLIFIED ALERT system.

        ULTRA-SIMPLIFIED LOGIC:
        ======================
        Only 2 fields: status and is_alert_triggered
        No "alert" key - it's redundant!

        STATES:
        =======
        1. Normal: status="", is_alert_triggered=False
        2. Alert (First Time): status="High Occupancy", is_alert_triggered=True
        3. Alert (Already Sent): status="", is_alert_triggered=False

        LOGIC:
        ======
        if current >= alert_people:
            if not already_sent:
                status = "High Occupancy"
                is_alert_triggered = True (SEND NOTIFICATION)
            else:
                status = ""
                is_alert_triggered = False (SUPPRESS - already sent)
        else:
            status = ""
            is_alert_triggered = False (NORMAL - clear state)

        Args:
            people (List[Dict[str, Any]]): List of detected people
            threshold (int): Maximum capacity
            alert_rate (int): Alert percentage (0-100)

        Returns:
            Dict[str, Any]: Metrics dictionary with:
                - current_occupancy: Number of people
                - occupancy_percentage: Occupancy as percentage
                - over_capacity_count: People over threshold
                - avg_dwell_time: Average time people have been present
                - status: "High Occupancy" or ""
                - is_alert_triggered: True only for NEW alerts
                - net_count: total_entries - total_exits
        """
        current_occupancy = len(people)
        occupancy_percentage = (current_occupancy / threshold * 100.0) if threshold > 0 else 0.0
        over_capacity_count = max(0, current_occupancy - threshold)

        # Calculate average dwell time
        if len(people) > 0:
            # People are present - calculate average of currently present people
            dwell_times = [p.get("dwell_time", 0.0) for p in people if p.get("dwell_time", 0.0) > 0]
            avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0
            logger.debug(f"Camera {self.camera_id}: 👥 People present ({len(people)}), "
                         f"avg_dwell_time = {avg_dwell_time:.2f} (from current people)")
        else:
            # No people present - use dwell_60_time (mean of top 10 from last 60 minutes)
            avg_dwell_time = self.dwell_60_time
            logger.info(f"Camera {self.camera_id}: 📊 No people present, "
                        f"avg_dwell_time = {avg_dwell_time:.2f} (from dwell_60_time, "
                        f"top_10_dwell has {len(self.top_10_dwell)} items)")

        # ============= THRESHOLD CALCULATION =============
        alert_people = round(threshold * (alert_rate / 100.0))

        logger.debug(f"Camera {self.camera_id}: Alert threshold - "
                     f"{alert_people} people ({alert_rate}%)")

        # DEBUG: Log current state for troubleshooting
        logger.debug(f"🔍 Camera {self.camera_id} Alert Debug:")
        logger.debug(f"   Current occupancy: {current_occupancy} people")
        logger.debug(f"   Threshold: {threshold} people")
        logger.debug(f"   Alert rate: {alert_rate}%")
        logger.debug(f"   Alert triggers at: {alert_people} people")
        logger.debug(f"   Critical alert state: {self.critical_alert_state}")

        # ============= SIMPLIFIED ALERT LOGIC =============
        # Check if we're in alert condition
        is_in_alert_condition = current_occupancy >= alert_people

        logger.debug(f"   Is in alert condition: {is_in_alert_condition} ({current_occupancy} >= {alert_people})")

        # Manage status and is_alert_triggered
        status = ""
        is_alert_triggered = False

        if is_in_alert_condition and not self.critical_alert_state["critical_alert_active"]:
            # STATE 1: NEW ALERT - First time reaching alert threshold
            status = "High Occupancy"
            is_alert_triggered = True
            self.critical_alert_state["critical_alert_active"] = True
            self.critical_alert_state["first_triggered_frame"] = self.frame_count

            logger.info(f"🚨 Camera {self.camera_id}: ALERT TRIGGERED - "
                        f"{current_occupancy} people >= {alert_people} people "
                        f"({occupancy_percentage:.1f}% >= {alert_rate}%) - "
                        f"status='High Occupancy', is_alert_triggered=True")

        elif is_in_alert_condition and self.critical_alert_state["critical_alert_active"]:
            # STATE 2: ALREADY ALERTED - Suppress to avoid spam
            status = ""
            is_alert_triggered = False

            logger.debug(f"Camera {self.camera_id}: Alert SUPPRESSED - "
                         f"{current_occupancy} people >= {alert_people}, "
                         f"already sent - status='', is_alert_triggered=False")

        elif not is_in_alert_condition and self.critical_alert_state["critical_alert_active"]:
            # STATE 3: CLEAR ALERT STATE - Below threshold, reset
            self.critical_alert_state["critical_alert_active"] = False
            self.critical_alert_state["first_triggered_frame"] = None

            logger.info(f"✅ Camera {self.camera_id}: Alert CLEARED - "
                        f"{current_occupancy} people < {alert_people} people "
                        f"({occupancy_percentage:.1f}% < {alert_rate}%) - "
                        f"status='', is_alert_triggered=False")

            status = ""
            is_alert_triggered = False
        else:
            # NORMAL STATE - No alert condition
            status = ""
            is_alert_triggered = False

        return {
            "current_occupancy": current_occupancy,
            "occupancy_percentage": round(occupancy_percentage, 1),
            "over_capacity_count": over_capacity_count,
            "avg_dwell_time": round(avg_dwell_time, 2),
            "status": status,  # "High Occupancy" or ""
            "is_alert_triggered": is_alert_triggered,  # True only for NEW alerts
            "net_count": max(0, self.total_entries - self.total_exits),
        }

    def _extract_coordinates(self, people: List[Dict[str, Any]], height: int, width: int) -> Dict[str, List[float]]:
        """
        Extract center coordinates and dimensions from TRACKED PEOPLE bounding boxes.

        CRITICAL FIX: This method now extracts coordinates from tracked people,
        not raw YOLO detections. This ensures consistency:
        - If people=[], then x=[], y=[], w=[], h=[]
        - If people=[3 people], then x=[3], y=[3], w=[3], h=[3]

        Args:
            people (List[Dict[str, Any]]): List of tracked people with 'bbox' key
            height (int): Frame height in pixels
            width (int): Frame width in pixels

        Returns:
            Dict[str, List[float]]: Dictionary with:
                - x: Center x coordinates [175.0, 365.0, ...]
                - y: Center y coordinates [300.0, 300.0, ...]
                - w: Box widths [150.0, 130.0, ...]
                - h: Box heights [300.0, 320.0, ...]

        Note:
            Coordinates are clamped to frame boundaries [0, width/height]
        """
        coords = {"x": [], "y": [], "w": [], "h": []}

        for person in people:
            bbox = person.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox

                # Calculate center and dimensions
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                box_w = x2 - x1
                box_h = y2 - y1

                # Clamp to frame boundaries
                coords["x"].append(max(0.0, min(float(width), center_x)))
                coords["y"].append(max(0.0, min(float(height), center_y)))
                coords["w"].append(max(0.0, box_w))
                coords["h"].append(max(0.0, box_h))

        return coords

    def _get_exit_times(self) -> List[str]:
        """
        Get list of recent exit timestamps (last 10 exits).

        Returns:
            List[str]: ISO formatted timestamps of recent exits

        Example:
            ["2024-12-17T10:30:45.123456Z", "2024-12-17T10:31:20.789012Z", ...]
        """
        try:
            recent = list(self.recent_exits)[-10:]
            return [exit_rec.get("exit_timestamp", "") for exit_rec in recent]
        except Exception as e:
            logger.debug(f"Failed to get exit times: {e}")
            return []

    def _get_exit_ids(self) -> List[int]:
        """
        Get list of recent exit person IDs (last 10 exits).

        Returns:
            List[int]: Person IDs of recent exits [5, 12, 23, ...]
        """
        try:
            recent = list(self.recent_exits)[-10:]
            return [exit_rec.get("person_id", 0) for exit_rec in recent]
        except Exception as e:
            logger.debug(f"Failed to get exit IDs: {e}")
            return []

    def _annotate_frame(self, frame: np.ndarray, people: List[Dict[str, Any]],
                        threshold: int, occupancy_percentage: float,
                        status: str, avg_dwell_time: float = 0.0) -> np.ndarray:
        """
        Annotate frame with simplified visual style.

        Args:
            frame (np.ndarray): Original frame
            people (List[Dict[str, Any]]): Detected people with bboxes
            threshold (int): Maximum capacity
            occupancy_percentage (float): Current occupancy percentage
            status (str): Status field value ("High Occupancy" or "")
            avg_dwell_time (float): Average dwell time in seconds (from metrics)

        Returns:
            np.ndarray: Annotated frame with:
                - Deep green bounding boxes
                - Red ID text (centered in box)
                - Deep blue timer (top of box)
                - Info panel with semi-transparent overlay
                - Occupancy & Alert section with colored status box

        STYLE:
            - Bounding box: Deep green (0, 100, 0)
            - ID text: Red (0, 0, 255) with white background
            - Timer text: Deep blue (139, 0, 0) with white background
            - Alert box: Red for alert, Green for normal
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        try:
            # Calculate metrics
            total_people = len(people)
            # Use the passed avg_dwell_time from metrics (includes dwell_60_time logic)
            # NO LONGER calculating from people list here!

            # Convert average dwell time to HH:MM:SS format
            avg_dwell_hhmmss = seconds_to_hhmmss(avg_dwell_time)

            # Determine if alert is active (based on status)
            is_alert = (status == "High Occupancy")

            # ===== Info Panel Section (Top) =====
            info_y_start = 30
            line_height = 35
            total_lines = 3  # Reduced from 5
            font_scale = 0.8
            font_thickness = 2
            text_color = (255, 255, 255)  # White

            # Semi-transparent dark overlay for info panel
            overlay = annotated.copy()
            overlay_height = info_y_start + (line_height * total_lines) + 20
            cv2.rectangle(overlay, (10, 10), (450, overlay_height), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            # Info texts with HH:MM:SS format and 60min window note
            info_texts = [
                f"Total People: {total_people}",
                f"Avg Dwell Time (60min): {avg_dwell_hhmmss}",  # NEW: HH:MM:SS format with window indicator
                f"Entry: {self.total_entries} | Exit: {self.total_exits}"
            ]

            for i, text in enumerate(info_texts):
                y_pos = info_y_start + (i * line_height)
                cv2.putText(annotated, text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

            # ===== Occupancy & Alert Section (Below Other Info) =====
            separator_y = info_y_start + (line_height * total_lines) + 10

            # Draw separator line
            cv2.line(annotated, (20, separator_y), (440, separator_y), (200, 200, 200), 2)

            # Occupancy section start
            occ_section_y = separator_y + 20

            # Occupancy text
            occupancy_text = f"Occupancy: {occupancy_percentage:.1f}%"
            occ_font_scale = 1.0
            occ_font_thickness = 2

            cv2.putText(annotated, occupancy_text, (20, occ_section_y),
                        cv2.FONT_HERSHEY_SIMPLEX, occ_font_scale, text_color, occ_font_thickness)

            # Alert status box below occupancy (only show when alert is active)
            if is_alert:
                alert_y = occ_section_y + 40
                alert_text = "ALERT!"
                alert_font_scale = 1.2
                alert_font_thickness = 3

                # Red box for alert
                alert_box_x1 = 20
                alert_box_y1 = alert_y - 30
                alert_box_x2 = 200
                alert_box_y2 = alert_y + 10

                alert_bg_color = (0, 0, 255)  # Red
                alert_text_color = (255, 255, 255)  # White text

                # Draw alert status box
                cv2.rectangle(annotated, (alert_box_x1, alert_box_y1), (alert_box_x2, alert_box_y2),
                              alert_bg_color, -1)

                # Draw alert text centered in box
                alert_text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                  alert_font_scale, alert_font_thickness)[0]
                alert_text_x = alert_box_x1 + ((alert_box_x2 - alert_box_x1) - alert_text_size[0]) // 2
                alert_text_y = alert_box_y1 + ((alert_box_y2 - alert_box_y1) + alert_text_size[1]) // 2

                cv2.putText(annotated, alert_text, (alert_text_x, alert_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, alert_font_scale, alert_text_color, alert_font_thickness)

            # ===== Draw bounding boxes and labels =====
            for person in people:
                try:
                    box = person["bbox"]
                    person_id = person["id"]
                    dwell_time = person.get("dwell_time", 0.0)

                    x1, y1, x2, y2 = [int(coord) for coord in box]

                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y2 = max(y1 + 1, min(y2, height))

                    # Deep green bounding box
                    deep_green = (0, 100, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), deep_green, 3)

                    # Red ID text (centered in box)
                    red_color = (0, 0, 255)
                    id_text = f"ID: {person_id}"

                    text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = x1 + (x2 - x1 - text_size[0]) // 2
                    text_y = y1 + (y2 - y1 + text_size[1]) // 2

                    text_x = max(x1 + 5, min(text_x, x2 - text_size[0] - 5))
                    text_y = max(y1 + text_size[1] + 5, min(text_y, y2 - 5))

                    cv2.rectangle(annotated,
                                  (text_x - 5, text_y - text_size[1] - 5),
                                  (text_x + text_size[0] + 5, text_y + 5),
                                  (255, 255, 255), -1)

                    cv2.putText(annotated, id_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)

                    # Deep blue timer in HH:MM:SS format (top of box)
                    deep_blue = (139, 0, 0)
                    timer_text = seconds_to_hhmmss(dwell_time)  # NEW: HH:MM:SS format

                    timer_y = max(y1 - 10, 25)
                    timer_x = x1

                    timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated,
                                  (timer_x - 2, timer_y - timer_size[1] - 5),
                                  (timer_x + timer_size[0] + 2, timer_y + 5),
                                  (255, 255, 255), -1)

                    cv2.putText(annotated, timer_text, (timer_x, timer_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, deep_blue, 2)

                except Exception as e:
                    logger.debug(f"Annotation failed for person {person.get('id', 'unknown')}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Frame annotation failed for camera {self.camera_id}: {e}")

        return annotated

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert frame to base64-encoded JPEG string.

        Args:
            frame (np.ndarray): Frame to encode

        Returns:
            str: Base64-encoded JPEG string (or empty string on failure)
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return ""

    def get_camera_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive camera statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary with:
                - camera_id: Camera identifier
                - frames_processed: Total frames processed
                - total_entries: Lifetime entry count
                - total_exits: Lifetime exit count
                - net_count: entries - exits
                - active_people_count: Currently present people
                - critical_alert_state: Current alert state
        """
        return {
            "camera_id": self.camera_id,
            "frames_processed": self.frame_count,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "net_count": max(0, self.total_entries - self.total_exits),
            "active_people_count": len(self.active_people),
            "critical_alert_state": self.critical_alert_state.copy()
        }


# ============================================================================
# PEOPLE COUNTING SYSTEM MANAGER
# ============================================================================

class PeopleCountingSystemManager:
    """
    Manager for multiple camera counting systems.

    This class:
    - Creates and manages camera-specific counting systems
    - Provides unified interface for multi-camera setup
    - Handles dynamic camera addition
    - Thread-safe operations

    Attributes:
        yolo_model: Shared YOLO model instance
        config (Dict[str, Any]): Shared configuration
        camera_systems (Dict[int, CameraPeopleCountingSystem]): Camera instances
    """

    def __init__(self, yolo_model, config: Dict[str, Any]):
        """
        Initialize the system manager.

        Args:
            yolo_model: Pre-loaded YOLO model (shared across all cameras)
            config (Dict[str, Any]): Configuration dictionary
        """
        self.yolo_model = yolo_model
        self.config = config
        self.camera_systems: Dict[int, CameraPeopleCountingSystem] = {}
        self._lock = threading.Lock()

        logger.info("PeopleCountingSystemManager initialized with "
                    "SIMPLIFIED DUAL STATUS/ALERT system (warning fixed at 65%)")

    def get_or_create_system(self, camid: int) -> CameraPeopleCountingSystem:
        """
        Get existing counting system or create new one for camera.

        Args:
            camid (int): Camera identifier

        Returns:
            CameraPeopleCountingSystem: Camera-specific counting system

        Note:
            Thread-safe: Multiple threads can call this simultaneously
        """
        with self._lock:
            if camid not in self.camera_systems:
                logger.info(f"Creating new counting system for camera {camid}")
                self.camera_systems[camid] = CameraPeopleCountingSystem(
                    camera_id=camid,
                    yolo_model=self.yolo_model,
                    config=self.config
                )
            return self.camera_systems[camid]

    def process_frame(self, frame: np.ndarray, camid: int, org_id: int, userid: int,
                      threshold: int, alert_rate: int, return_annotated: bool = False,
                      confidence_threshold: float = 0.35) -> Dict[str, Any]:
        """
        Process frame with camera-specific system.

        This is the main entry point for the manager. It:
        1. Gets or creates camera-specific system
        2. Updates confidence threshold if changed
        3. Processes frame
        4. Adds org_id and userid to response

        Args:
            frame (np.ndarray): Input frame
            camid (int): Camera identifier
            org_id (int): Organization identifier
            userid (int): User identifier
            threshold (int): Maximum capacity
            alert_rate (int): Critical alert percentage
            return_annotated (bool): Whether to return annotated frame
            confidence_threshold (float): YOLO confidence threshold

        Returns:
            Dict[str, Any]: Complete response with all metrics
        """
        # Get or create camera-specific system
        system = self.get_or_create_system(camid)

        # Update confidence threshold if changed
        if abs(system.confidence_threshold - confidence_threshold) > 0.001:
            system.set_confidence_threshold(confidence_threshold)

        # Process frame (warning at fixed 65%)
        result = system.process_frame(frame, threshold, alert_rate, return_annotated)

        # Add org_id and userid to response
        result["org_id"] = org_id
        result["userid"] = userid

        return result

    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for all cameras.

        Returns:
            Dict[int, Dict[str, Any]]: Statistics per camera {camid: stats}
        """
        with self._lock:
            return {camid: system.get_camera_stats()
                    for camid, system in self.camera_systems.items()}