# people_counting.py
"""
People Counting System - Version 2
Fixed camera-specific state isolation and robust tracking
Optimized for SageMaker ml.g4dn.xlarge with enhanced annotations
"""

import time
import base64
import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_timestamp() -> str:
    """Get UTC timestamp"""
    return datetime.now(timezone.utc).isoformat()

@dataclass
class Track:
    """Track state for robust object tracking"""
    id: int
    bbox: List[float]
    confidence: float
    features: np.ndarray
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    state: str = "tentative"
    hit_streak: int = 0
    time_since_update: int = 0
    entry_time: float = 0.0
    alert=True
    
    def predict(self):
        """Simple motion prediction"""
        new_center = (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1]
        )
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = [
            new_center[0] - w/2,
            new_center[1] - h/2,
            new_center[0] + w/2,
            new_center[1] + h/2
        ]
        self.center = new_center
        self.time_since_update += 1
    
    def update(self, bbox: List[float], confidence: float, features: np.ndarray):
        """Update track with new detection"""
        old_center = self.center
        new_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        
        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        
        self.bbox = bbox
        self.confidence = confidence
        self.features = features
        self.center = new_center
        self.hit_streak += 1
        self.time_since_update = 0
        
        if self.state == "tentative" and self.hit_streak >= 3:
            self.state = "confirmed"
        elif self.state == "lost" and self.hit_streak >= 1:
            self.state = "confirmed"

class RobustReIdentifier:
    """Enhanced visual re-identification with multiple features"""
    
    def __init__(self, max_stored: int = 20, similarity_threshold: float = 0.65):
        self.max_stored = max_stored
        self.similarity_threshold = similarity_threshold
        self.stored_features = {}
        self._lock = threading.Lock()
        self.alert=True
        
    def extract_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract multi-modal features (color + texture)"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            h, w = frame.shape[:2]
            
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return np.zeros(128, dtype=np.float32)
            
            crop_resized = cv2.resize(crop, (64, 128))
            
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            texture_hist = np.histogram(magnitude, bins=16, range=(0, 255))[0]
            
            color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            texture_features = texture_hist.astype(np.float32)
            
            features = np.concatenate([color_features, texture_features])
            
            if np.sum(features) > 0:
                features = features / (np.linalg.norm(features) + 1e-6)
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return np.zeros(128, dtype=np.float32)
    
    def calculate_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate weighted similarity"""
        try:
            color_sim = np.dot(feat1[:48], feat2[:48]) / (
                np.linalg.norm(feat1[:48]) * np.linalg.norm(feat2[:48]) + 1e-6)
            
            texture_sim = np.dot(feat1[48:], feat2[48:]) / (
                np.linalg.norm(feat1[48:]) * np.linalg.norm(feat2[48:]) + 1e-6)
            
            return 0.7 * color_sim + 0.3 * texture_sim
        except:
            return 0.0
    
    def find_match(self, features: np.ndarray, current_time: float, 
                   current_location: Tuple[float, float]) -> Optional[str]:
        """Find matching person with spatial and temporal constraints"""
        with self._lock:
            best_match = None
            best_similarity = 0.0
            
            to_remove = []
            for person_id, (_, timestamp, _) in self.stored_features.items():
                if current_time - timestamp > 600:
                    to_remove.append(person_id)
            
            for person_id in to_remove:
                del self.stored_features[person_id]
            
            for person_id, (stored_features, timestamp, exit_location) in self.stored_features.items():
                time_diff = current_time - timestamp
                if time_diff > 300:
                    continue
                
                if exit_location:
                    spatial_distance = np.sqrt(
                        (current_location[0] - exit_location[0])**2 + 
                        (current_location[1] - exit_location[1])**2
                    )
                    if spatial_distance > 200:
                        continue
                
                similarity = self.calculate_similarity(features, stored_features)
                
                temporal_weight = max(0.5, 1.0 - time_diff / 300)
                weighted_similarity = similarity * temporal_weight
                
                if weighted_similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = weighted_similarity
                    best_match = person_id
            
            if best_match:
                del self.stored_features[best_match]
                logger.info(f"Re-identified person {best_match} (similarity: {best_similarity:.3f})")
            
            return best_match
    
    def store_features(self, person_id: str, features: np.ndarray, 
                      timestamp: float, exit_location: Tuple[float, float]):
        """Store features when person exits"""
        with self._lock:
            self.stored_features[person_id] = (features, timestamp, exit_location)
            
            if len(self.stored_features) > self.max_stored:
                oldest_id = min(self.stored_features.keys(), 
                              key=lambda k: self.stored_features[k][1])
                del self.stored_features[oldest_id]

class RobustTracker:
    """Robust multi-object tracker with state management"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        self.next_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._lock = threading.Lock()
    
    def update(self, detections: List[List[float]], 
               confidences: List[float], features_list: List[np.ndarray]) -> Dict[int, Track]:
        """Update tracker with new detections"""
        with self._lock:
            for track in self.tracks.values():
                if track.state in ["confirmed", "lost"]:
                    track.predict()
            
            if detections:
                matched, unmatched_dets, unmatched_trks = self._associate(
                    detections, confidences, features_list)
                
                for det_idx, trk_id in matched:
                    self.tracks[trk_id].update(
                        detections[det_idx], confidences[det_idx], features_list[det_idx])
                
                for det_idx in unmatched_dets:
                    self._create_track(detections[det_idx], confidences[det_idx], features_list[det_idx])
                
                for trk_id in unmatched_trks:
                    track = self.tracks[trk_id]
                    if track.state == "confirmed":
                        track.state = "lost"
                    track.hit_streak = 0
            
            to_delete = []
            for trk_id, track in self.tracks.items():
                if track.time_since_update > self.max_disappeared:
                    to_delete.append(trk_id)
                elif track.state == "tentative" and track.time_since_update > 3:
                    to_delete.append(trk_id)
            
            for trk_id in to_delete:
                del self.tracks[trk_id]
            
            return {tid: track for tid, track in self.tracks.items() 
                   if track.state in ["confirmed", "lost"]}
    
    def _associate(self, detections: List[List[float]], confidences: List[float], 
                   features_list: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using IoU and appearance"""
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        
        for det_idx, (det_bbox, det_conf, det_feat) in enumerate(zip(detections, confidences, features_list)):
            det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
            
            for trk_idx, trk_id in enumerate(track_ids):
                track = self.tracks[trk_id]
                
                spatial_dist = np.sqrt((det_center[0] - track.center[0])**2 + 
                                     (det_center[1] - track.center[1])**2)
                spatial_cost = min(1.0, spatial_dist / self.max_distance)
                
                appearance_sim = self._appearance_similarity(det_feat, track.features)
                appearance_cost = 1.0 - appearance_sim
                
                total_cost = 0.6 * spatial_cost + 0.4 * appearance_cost
                cost_matrix[det_idx, trk_idx] = total_cost
        
        matched, unmatched_dets, unmatched_trks = self._greedy_assignment(
            cost_matrix, track_ids, threshold=0.7)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _greedy_assignment(self, cost_matrix: np.ndarray, track_ids: List[int], 
                          threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy assignment approximation"""
        matched = []
        used_det_indices = set()
        used_trk_indices = set()
        
        det_indices, trk_indices = np.where(cost_matrix < threshold)
        costs = cost_matrix[det_indices, trk_indices]
        sorted_indices = np.argsort(costs)
        
        for idx in sorted_indices:
            det_idx = det_indices[idx]
            trk_idx = trk_indices[idx]
            
            if det_idx not in used_det_indices and trk_idx not in used_trk_indices:
                matched.append((det_idx, track_ids[trk_idx]))
                used_det_indices.add(det_idx)
                used_trk_indices.add(trk_idx)
        
        unmatched_dets = [i for i in range(cost_matrix.shape[0]) if i not in used_det_indices]
        unmatched_trks = [track_ids[i] for i in range(len(track_ids)) if i not in used_trk_indices]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _appearance_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate appearance similarity"""
        try:
            return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)
        except:
            return 0.0
    
    def _create_track(self, bbox: List[float], confidence: float, features: np.ndarray):
        """Create new track"""
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        track = Track(
            id=self.next_id,
            bbox=bbox,
            confidence=confidence,
            features=features,
            center=center,
            entry_time=time.time()
        )
        self.tracks[self.next_id] = track
        self.next_id += 1

class CameraPeopleCountingSystem:
    """Camera-specific people counting system with isolated state"""
    
    def __init__(self, camera_id: int, yolo_model, config: Dict[str, Any]):
        self.camera_id = camera_id
        self.yolo_model = yolo_model
        self.config = config or {}
        self.alert=True
        
        self.confidence_threshold = self.config.get("confidence_threshold", 0.35)
        
        self.tracker = RobustTracker(max_disappeared=30, max_distance=100.0)
        self.reidentifier = RobustReIdentifier(
            max_stored=self.config.get("max_stored_features", 20),
            similarity_threshold=self.config.get("similarity_threshold", 0.65)
        )
        
        self.frame_count = 0
        self.total_entries = 0
        self.total_exits = 0
        self.active_people = {}
        
        self.entry_log = deque(maxlen=200)
        self.exit_log = deque(maxlen=200)
        
        self._lock = threading.Lock()
        
        logger.info(f"Camera {camera_id} counting system initialized - confidence: {self.confidence_threshold}")
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.01, min(0.99, float(threshold)))
        logger.info(f"Camera {self.camera_id}: Updated confidence threshold to {self.confidence_threshold}")
    
    def detect_people(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """Detect people using YOLO with enhanced debugging"""
        if frame is None or frame.size == 0:
            logger.warning(f"Camera {self.camera_id}: Invalid frame input")
            return [], []
        
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"Camera {self.camera_id}: Running YOLO prediction with conf={self.confidence_threshold}")
            
            results = self.yolo_model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                classes=[0] , # Person class only
                batch=8
            )
            
            boxes = []
            confidences = []
            
            if results and len(results) > 0:
                result = results[0]
                logger.debug(f"Camera {self.camera_id}: YOLO returned {len(results)} result(s)")
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    # Check if classes exist
                    classes = None
                    if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                        classes = result.boxes.cls.cpu().numpy()
                    
                    logger.info(f"Camera {self.camera_id}: Raw YOLO detections: {len(boxes_xyxy)}")
                    logger.info(f"Camera {self.camera_id}: Confidence threshold: {self.confidence_threshold}")
                    
                    for i, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
                        x1, y1, x2, y2 = box
                        cls = classes[i] if classes is not None else 0
                        
                        logger.debug(f"Detection {i}: conf={conf:.3f}, class={cls}, box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                        
                        # Validation checks
                        valid_box = x2 > x1 and y2 > y1
                        valid_conf = conf >= self.confidence_threshold
                        valid_class = cls == 0  # Person class
                        
                        if valid_box and valid_conf and valid_class:
                            boxes.append([float(x1), float(y1), float(x2), float(y2)])
                            confidences.append(float(conf))
                            logger.debug(f"✓ Accepted detection {i}")
                        else:
                            reject_reason = []
                            if not valid_box:
                                reject_reason.append("invalid_box")
                            if not valid_conf:
                                reject_reason.append(f"low_conf({conf:.3f}<{self.confidence_threshold})")
                            if not valid_class:
                                reject_reason.append(f"wrong_class({cls})")
                            logger.debug(f"✗ Rejected detection {i}: {','.join(reject_reason)}")
                    
                    logger.info(f"Camera {self.camera_id}: Final accepted detections: {len(boxes)}")
                else:
                    logger.warning(f"Camera {self.camera_id}: YOLO result has no boxes attribute")
            else:
                logger.warning(f"Camera {self.camera_id}: No YOLO results or empty results")
            
            return boxes, confidences
            
        except Exception as e:
            logger.error(f"Detection failed for camera {self.camera_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], []
    
    def process_frame(self, frame: np.ndarray, threshold: int, alert_rate: float, 
                     return_annotated: bool = False) -> Dict[str, Any]:
        """Process frame with camera-specific state management"""
        with self._lock:
            self.frame_count += 1
            current_time = time.time()
            timestamp = get_timestamp()
            frame_id = f"FR_{self.camera_id}_{int(current_time * 1000)}"
            
            logger.info(f"Camera {self.camera_id}: Processing frame {self.frame_count}")
            
            try:
                height, width = frame.shape[:2]
                logger.debug(f"Camera {self.camera_id}: Frame size: {width}x{height}")
                
                # Detect people
                boxes, confidences = self.detect_people(frame)
                logger.info(f"Camera {self.camera_id}: Detected {len(boxes)} people")
                
                # Extract features for re-identification
                features_list = []
                for i, box in enumerate(boxes):
                    features = self.reidentifier.extract_features(frame, box)
                    features_list.append(features)
                    logger.debug(f"Camera {self.camera_id}: Extracted features for detection {i}")
                
                # Update tracker
                tracks = self.tracker.update(boxes, confidences, features_list)
                logger.info(f"Camera {self.camera_id}: Active tracks: {len(tracks)}")
                
                # Process tracks and handle entries/exits
                people = self._process_tracks(tracks, current_time)
                logger.info(f"Camera {self.camera_id}: Final people count: {len(people)}")
                
                # Calculate metrics
                metrics = self._calculate_metrics(people, threshold, alert_rate)
                
                # Extract coordinates
                coords = self._extract_coordinates(boxes, height, width)
                
                # Annotated frame
                annotated_frame_b64 = None
                if return_annotated:
                    try:
                        annotated = self._annotate_frame(frame, people)
                        annotated_frame_b64 = self._frame_to_base64(annotated)
                    except Exception as e:
                        logger.debug(f"Annotation failed for camera {self.camera_id}: {e}")
                
                # Build response
                response = {
                    "camid": self.camera_id,
                    "Frame_Id": frame_id,
                    "Time_stamp": timestamp,
                    "Frame_Count": self.frame_count,
                    "Total_people_detected": len(people),
                    "Current_occupancy": metrics["current_occupancy"],
                    "People_ids": [p["id"] for p in people],
                    "Entry_time": [p.get("entry_time", 0.0) for p in people],
                    "People_dwell_time": [p.get("dwell_time", 0.0) for p in people],
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
                    "Average_dwell_time": metrics["avg_dwell_time"],
                    "Max_occupancy": threshold,
                    "Status": metrics["status"],
                    "is_alert_triggered": metrics["is_alert_triggered"],
                    "Processing_Status": 1,
                    "annotated_frame": annotated_frame_b64
                }
                
                logger.info(f"Camera {self.camera_id} Frame {self.frame_count}: {len(people)} people, "
                           f"occupancy: {metrics['occupancy_percentage']:.1f}%")
                
                return response
                
            except Exception as e:
                logger.error(f"Frame processing failed for camera {self.camera_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                return {
                    "camid": self.camera_id,
                    "Frame_Id": frame_id,
                    "Time_stamp": timestamp,
                    "Frame_Count": self.frame_count,
                    "Total_people_detected": 0,
                    "Current_occupancy": 0,
                    "People_ids": [],
                    "Entry_time": [],
                    "People_dwell_time": [],
                    "Confidence_scores": [],
                    "Bounding_boxes": [],
                    "x": [], "y": [], "w": [], "h": [],
                    "accuracy": [],
                    "Total_entries": self.total_entries,
                    "Total_exits": self.total_exits,
                    "Net_count": max(0, self.total_entries - self.total_exits),
                    "Occupancy_percentage": 0.0,
                    "Average_dwell_time": 0.0,
                    "Max_occupancy": threshold,
                    "Status": "Error",
                    "is_alert_triggered": False,
                    "Processing_Status": 0,
                    "error_message": str(e)[:200]
                }
    
    def _process_tracks(self, tracks: Dict[int, Track], current_time: float) -> List[Dict[str, Any]]:
        """Process tracks and handle entries/exits"""
        people = []
        current_track_ids = set(tracks.keys())
        previous_track_ids = set(self.active_people.keys())
        
        logger.debug(f"Camera {self.camera_id}: Current tracks: {current_track_ids}")
        logger.debug(f"Camera {self.camera_id}: Previous tracks: {previous_track_ids}")
        
        new_track_ids = current_track_ids - previous_track_ids
        for track_id in new_track_ids:
            track = tracks[track_id]
            
            features = track.features
            center = track.center
            matched_id = self.reidentifier.find_match(features, current_time, center)
            
            if matched_id:
                person_id = matched_id
                logger.info(f"Camera {self.camera_id}: Re-entry of person {person_id}")
            else:
                self.total_entries += 1
                person_id = f"P{self.camera_id}_{self.total_entries}"
                
                self.entry_log.append({
                    "id": person_id,
                    "track_id": track_id,
                    "time": current_time,
                    "timestamp": get_timestamp()
                })
                
                logger.info(f"Camera {self.camera_id}: New person entry {person_id}")
            
            self.active_people[track_id] = {
                "person_id": person_id,
                "entry_time": current_time,
                "features": features
            }
        
        exited_track_ids = previous_track_ids - current_track_ids
        for track_id in exited_track_ids:
            person_data = self.active_people[track_id]
            person_id = person_data["person_id"]
            entry_time = person_data["entry_time"]
            features = person_data["features"]
            
            exit_location = (0, 0)
            self.reidentifier.store_features(person_id, features, current_time, exit_location)
            
            time_spent = current_time - entry_time
            self.exit_log.append({
                "id": person_id,
                "track_id": track_id,
                "time": current_time,
                "time_spent": time_spent,
                "timestamp": get_timestamp()
            })
            
            self.total_exits += 1
            del self.active_people[track_id]
            
            logger.info(f"Camera {self.camera_id}: Person exit {person_id} (spent {time_spent:.1f}s)")
        
        for track_id, track in tracks.items():
            if track_id in self.active_people:
                person_data = self.active_people[track_id]
                entry_time = person_data["entry_time"]
                dwell_time = current_time - entry_time
                
                person = {
                    "id": person_data["person_id"],
                    "track_id": track_id,
                    "bbox": track.bbox,
                    "confidence": track.confidence,
                    "entry_time": entry_time,
                    "dwell_time": dwell_time,
                    "center": track.center
                }
                people.append(person)
        
        logger.debug(f"Camera {self.camera_id}: Processed {len(people)} people")
        return people
    
    def _calculate_metrics(self, people: List[Dict[str, Any]], threshold: int, 
                          alert_rate: float) -> Dict[str, Any]:
        """Calculate metrics"""
        current_occupancy = len(people)
        occupancy_percentage = (current_occupancy / threshold * 100.0) if threshold > 0 else 0.0
        occupancy_percentage = min(100.0, max(0.0, occupancy_percentage))
        
        dwell_times = [p.get("dwell_time", 0.0) for p in people if p.get("dwell_time", 0.0) > 0]
        avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0
        status=""
        if occupancy_percentage >= alert_rate and self.alert:    # .8 > 0.7
            status = "High Occupancy"
            self.alert=False
        elif occupancy_percentage >= alert_rate * 0.7  and self.alert:
            status = "Medium Occupancy"
            self.alert=False
        elif self.alert:
            status="low Occupancy"

        
        if occupancy_percentage<alert_rate and not self.alert:
            self.alert=True
        elif occupancy_percentage < alert_rate * 0.7  and self.alert:
            self.alert=True
        
        return {
            "current_occupancy": current_occupancy,
            "occupancy_percentage": round(occupancy_percentage, 1),
            "avg_dwell_time": round(avg_dwell_time, 2),
            "status": status,
            "is_alert_triggered": occupancy_percentage >= alert_rate,
            "net_count": max(0, self.total_entries - self.total_exits)
        }
    
    def _extract_coordinates(self, boxes: List[List[float]], height: int, width: int) -> Dict[str, List[float]]:
        """Extract coordinates"""
        coords = {"x": [], "y": [], "w": [], "h": []}
        
        for box in boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                box_w = x2 - x1
                box_h = y2 - y1
                
                coords["x"].append(max(0.0, min(float(width), center_x)))
                coords["y"].append(max(0.0, min(float(height), center_y)))
                coords["w"].append(max(0.0, box_w))
                coords["h"].append(max(0.0, box_h))
        
        return coords
    
    def _annotate_frame(self, frame: np.ndarray, people: List[Dict[str, Any]]) -> np.ndarray:
        """Enhanced frame annotations"""
        if frame is None:
            return frame
        
        annotated = frame.copy()
        
        try:
            total_people = len(people)
            dwell_times = [p.get("dwell_time", 0.0) for p in people if p.get("dwell_time", 0.0) > 0]
            avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0
            
            info_y_start = 30
            line_height = 35
            font_scale = 0.8
            font_thickness = 2
            text_color = (0, 0, 0)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (450, info_y_start + (line_height * 5) + 10), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            info_texts = [
                f"Camera ID: {self.camera_id}",
                f"Total People: {total_people}",
                f"Avg Dwell Time: {avg_dwell_time:.1f}s",
                f"Entry Count: {self.total_entries}",
                f"Exit Count: {self.total_exits}"
            ]
            
            for i, text in enumerate(info_texts):
                y_pos = info_y_start + (i * line_height)
                cv2.putText(annotated, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            
            for person in people:
                try:
                    box = person["bbox"]
                    person_id = person["id"]
                    dwell_time = person.get("dwell_time", 0.0)
                    
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    height, width = annotated.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y2 = max(y1 + 1, min(y2, height))
                    
                    deep_green = (0, 100, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), deep_green, 3)
                    
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
                    
                    deep_blue = (139, 0, 0)
                    timer_text = f"{dwell_time:.1f}s"
                    
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
    
    def _frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """Convert frame to base64"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
        except:
            return None

class PeopleCountingSystemManager:
    """Manager for multiple camera systems with proper state isolation"""
    
    def __init__(self, yolo_model, config: Dict[str, Any]):
        self.yolo_model = yolo_model
        self.config = config
        self.camera_systems = {}
        self._lock = threading.Lock()
        
        logger.info("People Counting System Manager initialized")
    
    def get_or_create_camera_system(self, camera_id: int) -> CameraPeopleCountingSystem:
        """Get or create camera-specific system"""
        with self._lock:
            if camera_id not in self.camera_systems:
                self.camera_systems[camera_id] = CameraPeopleCountingSystem(
                    camera_id, self.yolo_model, self.config)
                logger.info(f"Created new camera system for camera {camera_id}")
            
            return self.camera_systems[camera_id]
    
    def process_frame(self, frame: np.ndarray, camid: int, org_id: int, userid: int,
                     threshold: int, alert_rate: float, return_annotated: bool = False,
                     confidence_threshold: float = 0.35) -> Dict[str, Any]:
        """Process frame with camera-specific system"""
        camera_system = self.get_or_create_camera_system(camid)
        
        camera_system.set_confidence_threshold(confidence_threshold)
        
        result = camera_system.process_frame(frame, threshold, alert_rate, return_annotated)
        
        result["org_id"] = org_id
        result["userid"] = userid
        
        return result
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all cameras"""
        with self._lock:
            total_entries = sum(sys.total_entries for sys in self.camera_systems.values())
            total_exits = sum(sys.total_exits for sys in self.camera_systems.values())
            total_current = sum(len(sys.active_people) for sys in self.camera_systems.values())
            
            return {
                "total_cameras": len(self.camera_systems),
                "total_entries": total_entries,
                "total_exits": total_exits,
                "total_current_occupancy": total_current,
                "cameras": list(self.camera_systems.keys())
            }

# Module-level exports for proper importing
__all__ = [
    'PeopleCountingSystemManager',
    'CameraPeopleCountingSystem', 
    'RobustTracker',
    'RobustReIdentifier',
    'Track',
    'get_timestamp'
]