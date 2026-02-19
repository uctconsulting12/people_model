"""
Optimized People Counting System with:
1. Fixed Hungarian matching bug (critical ID-switch fix)
2. Two-stage matching with hard gating
3. EMA feature aggregation for stability
4. Improved re-entry ReID with quality-aware exit features
5. TensorRT-ready architecture
6. Removed per-frame empty_cache() calls
"""

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes with constant velocity model.
    """
    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        self.kf = cv2.KalmanFilter(7, 4)  # 7 state vars, 4 measurements
        
        # State: [cx, cy, w, h, dx, dy, dw]
        # Measurement: [cx, cy, w, h]
        
        # Transition matrix (constant velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1
        
        # Initialize state
        cx, cy, w, h = self._bbox_to_state(bbox)
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0], dtype=np.float32)
        
    def _bbox_to_state(self, bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h
    
    def _state_to_bbox(self, state):
        """Convert [cx, cy, w, h, ...] to [x1, y1, x2, y2]"""
        cx, cy, w, h = state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        return self._state_to_bbox(self.kf.statePost)
    
    def update(self, bbox):
        """Update with measurement"""
        cx, cy, w, h = self._bbox_to_state(bbox)
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        self.kf.correct(measurement)
    
    def get_bbox(self):
        """Get current bbox estimate"""
        return self._state_to_bbox(self.kf.statePost)
    
    def get_center(self):
        """Get current center estimate"""
        cx, cy = self.kf.statePost[:2]
        return (float(cx), float(cy))


class Track:
    """
    Individual track with EMA feature aggregation and quality tracking.
    """
    def __init__(self, track_id, bbox, features, confidence):
        self.track_id = track_id
        self.bbox = bbox
        self.center = self._get_center(bbox)
        
        # Feature aggregation with EMA
        self.features = features.copy()  # Current EMA feature
        self.feature_alpha = 0.9  # EMA smoothing factor
        
        # Feature quality tracking
        self.feature_history = deque(maxlen=30)  # Last 30 frames
        self.feature_history.append({
            'features': features.copy(),
            'confidence': confidence,
            'bbox_area': self._bbox_area(bbox)
        })
        
        # Kalman filter
        self.kf = KalmanBoxTracker(bbox)
        
        # Track state
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'tentative'  # tentative, confirmed, lost
        
    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def predict(self):
        """Predict next position"""
        self.age += 1
        self.time_since_update += 1
        predicted_bbox = self.kf.predict()
        return predicted_bbox
    
    def update(self, bbox, features, confidence):
        """Update track with new detection"""
        self.bbox = bbox
        self.center = self._get_center(bbox)
        self.kf.update(bbox)
        
        # Update features with EMA
        self.features = self.feature_alpha * self.features + \
                       (1 - self.feature_alpha) * features
        # Normalize
        self.features = self.features / (np.linalg.norm(self.features) + 1e-8)
        
        # Store feature quality info
        self.feature_history.append({
            'features': features.copy(),
            'confidence': confidence,
            'bbox_area': self._bbox_area(bbox)
        })
        
        self.hits += 1
        self.time_since_update = 0
        
        # Update state
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed this frame"""
        self.time_since_update += 1
        if self.time_since_update > 5:
            self.state = 'lost'
    
    def get_best_exit_features(self):
        """
        Get best quality features from history for exit signature.
        Prioritizes: high confidence + large bbox + recent.
        """
        if not self.feature_history:
            return self.features.copy()
        
        # Score each frame by quality
        scored_features = []
        for frame_data in self.feature_history:
            score = frame_data['confidence'] * np.sqrt(frame_data['bbox_area'])
            scored_features.append((score, frame_data['features']))
        
        # Get top 10 and average them
        scored_features.sort(reverse=True, key=lambda x: x[0])
        top_features = [f for _, f in scored_features[:10]]
        
        if top_features:
            mean_features = np.mean(top_features, axis=0)
            mean_features = mean_features / (np.linalg.norm(mean_features) + 1e-8)
            return mean_features
        else:
            return self.features.copy()


class OptimizedPeopleCounter:
    """
    Optimized people counting with fixed matching and improved ReID.
    """
    def __init__(self, 
                 detector,
                 reid_model,
                 entry_line,
                 exit_line,
                 frame_width,
                 frame_height,
                 fps=30,
                 max_disappeared=30,
                 min_hits=3,
                 reentry_similarity_threshold=0.5):
        
        self.detector = detector
        self.reid_model = reid_model
        self.entry_line = entry_line
        self.exit_line = exit_line
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Tracking parameters (dynamically adjusted based on FPS)
        self.max_disappeared = int(max_disappeared * (fps / 30))  # Scale with FPS
        self.min_hits = min_hits
        self.reentry_similarity_threshold = reentry_similarity_threshold
        
        # Tracks
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        
        # Counting
        self.entry_count = 0
        self.exit_count = 0
        self.crossed_entry = set()
        self.crossed_exit = set()
        
        # Re-entry ReID
        self.exited_people = {}  # track_id -> exit_info
        self.active_people = set()  # Currently inside
        
        # Performance monitoring
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
    def process_frame(self, frame):
        """Process single frame"""
        import time
        start_time = time.time()
        
        # 1. Detect people
        detections = self.detector.detect(frame)
        
        # 2. Extract features for all detections (batched)
        detection_data = []
        if len(detections) > 0:
            crops = [self._crop_bbox(frame, det['bbox']) for det in detections]
            features_batch = self.reid_model.extract_features_batch(crops)
            
            for det, features in zip(detections, features_batch):
                detection_data.append({
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'features': features
                })
        
        # 3. Predict all tracks
        for track in self.tracks.values():
            track.predict()
        
        # 4. Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections(
            detection_data, list(self.tracks.keys()), self.tracks
        )
        
        # 5. Update matched tracks
        for det_idx, track_id in matched:
            det = detection_data[det_idx]
            self.tracks[track_id].update(
                det['bbox'], det['features'], det['confidence']
            )
        
        # 6. Handle unmatched detections (new tracks or re-entries)
        for det_idx in unmatched_dets:
            det = detection_data[det_idx]
            
            # Try re-entry matching
            reentry_id = self._match_reentry(det['features'], det['bbox'])
            
            if reentry_id is not None:
                # Re-entry detected
                self.tracks[reentry_id] = Track(
                    reentry_id, det['bbox'], det['features'], det['confidence']
                )
                self.tracks[reentry_id].state = 'confirmed'
                self.active_people.add(reentry_id)
                del self.exited_people[reentry_id]
            else:
                # New track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = Track(
                    new_id, det['bbox'], det['features'], det['confidence']
                )
                self.active_people.add(new_id)
        
        # 7. Handle unmatched tracks
        for track_id in unmatched_trks:
            self.tracks[track_id].mark_missed()
        
        # 8. Remove lost tracks and handle exits
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
                
                # Check if this is an exit
                if track.state == 'confirmed' and track_id in self.active_people:
                    exit_location = self._check_exit(track.center)
                    if exit_location:
                        # Store exit info with best features
                        self.exited_people[track_id] = {
                            'features': track.get_best_exit_features(),
                            'exit_location': exit_location,
                            'exit_bbox': track.bbox
                        }
        
        for track_id in tracks_to_remove:
            if track_id in self.active_people:
                self.active_people.remove(track_id)
            del self.tracks[track_id]
        
        # 9. Count line crossings
        self._update_counts()
        
        # Performance tracking
        self.frame_count += 1
        self.processing_times.append(time.time() - start_time)
        
        return self._get_results()
    
    def _associate_detections(self, detections, track_ids, tracks):
        """
        Two-stage association with hard gating.
        """
        if len(detections) == 0 or len(track_ids) == 0:
            return [], list(range(len(detections))), track_ids
        
        # Stage 1: High-confidence IoU matching
        stage1_matches, stage1_unmatched_dets, stage1_unmatched_trks = \
            self._match_stage1(detections, track_ids, tracks)
        
        # Stage 2: Appearance + spatial matching
        if len(stage1_unmatched_dets) > 0 and len(stage1_unmatched_trks) > 0:
            remaining_dets = [detections[i] for i in stage1_unmatched_dets]
            stage2_matches, stage2_unmatched_dets, stage2_unmatched_trks = \
                self._match_stage2(remaining_dets, stage1_unmatched_trks, tracks)
            
            # Map back to original indices
            stage2_matches = [
                (stage1_unmatched_dets[i], tid) for i, tid in stage2_matches
            ]
            final_unmatched_dets = [
                stage1_unmatched_dets[i] for i in stage2_unmatched_dets
            ]
            
            all_matches = stage1_matches + stage2_matches
            return all_matches, final_unmatched_dets, stage2_unmatched_trks
        else:
            return stage1_matches, stage1_unmatched_dets, stage1_unmatched_trks
    
    def _match_stage1(self, detections, track_ids, tracks):
        """Stage 1: IoU-based matching (high confidence)"""
        cost_matrix = np.full((len(detections), len(track_ids)), 1e5)
        
        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            
            for j, track_id in enumerate(track_ids):
                track = tracks[track_id]
                predicted_bbox = track.kf.get_bbox()
                iou = self._compute_iou(det_bbox, predicted_bbox)
                
                # Hard IoU gate
                if iou > 0.3:
                    cost_matrix[i, j] = 1 - iou
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        matched_det_indices = set()
        matched_trk_cols = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.7:
                matched.append((r, track_ids[c]))  # (det_idx, track_id)
                matched_det_indices.add(r)
                matched_trk_cols.add(c)
        
        unmatched_dets = [i for i in range(len(detections)) 
                         if i not in matched_det_indices]
        unmatched_trks = [track_ids[j] for j in range(len(track_ids)) 
                         if j not in matched_trk_cols]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _match_stage2(self, detections, track_ids, tracks):
        """Stage 2: Appearance + spatial matching"""
        if len(detections) == 0 or len(track_ids) == 0:
            return [], list(range(len(detections))), track_ids
        
        cost_matrix = np.full((len(detections), len(track_ids)), 1e5)
        
        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            det_features = det['features']
            det_center = self._get_center(det_bbox)
            
            for j, track_id in enumerate(track_ids):
                track = tracks[track_id]
                
                # Spatial cost using predicted center
                predicted_center = track.kf.get_center()
                spatial_dist = np.linalg.norm(
                    np.array(det_center) - np.array(predicted_center)
                )
                max_spatial = np.sqrt(self.frame_width**2 + self.frame_height**2)
                spatial_cost = spatial_dist / max_spatial
                
                # Appearance cost
                appearance_sim = self._cosine_similarity(det_features, track.features)
                appearance_cost = 1 - appearance_sim
                
                # Hard gates
                if spatial_cost > 0.5:  # Spatial gate
                    continue
                if appearance_sim < 0.3:  # Appearance gate
                    continue
                
                # Combined cost (favor appearance in stage 2)
                cost_matrix[i, j] = 0.3 * spatial_cost + 0.7 * appearance_cost
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        matched_det_indices = set()
        matched_trk_cols = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.6:  # Stricter threshold
                matched.append((r, track_ids[c]))
                matched_det_indices.add(r)
                matched_trk_cols.add(c)
        
        unmatched_dets = [i for i in range(len(detections)) 
                         if i not in matched_det_indices]
        unmatched_trks = [track_ids[j] for j in range(len(track_ids)) 
                         if j not in matched_trk_cols]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _match_reentry(self, features, bbox):
        """
        Match detection to exited person (re-entry).
        Uses stricter thresholds and spatial gating.
        """
        if not self.exited_people:
            return None
        
        det_center = self._get_center(bbox)
        best_match_id = None
        best_similarity = self.reentry_similarity_threshold
        
        for track_id, exit_info in self.exited_people.items():
            # Spatial gate: must be near exit location
            exit_location = exit_info['exit_location']
            dist_to_exit = np.linalg.norm(
                np.array(det_center) - np.array(exit_location)
            )
            max_reentry_dist = 200  # pixels
            
            if dist_to_exit > max_reentry_dist:
                continue
            
            # Appearance matching
            similarity = self._cosine_similarity(features, exit_info['features'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track_id
        
        return best_match_id
    
    def _update_counts(self):
        """Update entry/exit counts based on line crossings"""
        for track_id, track in self.tracks.items():
            if track.state != 'confirmed':
                continue
            
            center = track.center
            
            # Check entry crossing
            if track_id not in self.crossed_entry:
                if self._crosses_line(center, self.entry_line):
                    self.entry_count += 1
                    self.crossed_entry.add(track_id)
            
            # Check exit crossing
            if track_id not in self.crossed_exit:
                if self._crosses_line(center, self.exit_line):
                    self.exit_count += 1
                    self.crossed_exit.add(track_id)
    
    def _crosses_line(self, point, line):
        """Check if point crosses line"""
        # Simple implementation - can be enhanced with history
        (x1, y1), (x2, y2) = line
        px, py = point
        
        # Check if point is near line
        dist = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / \
               np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        return dist < 20  # threshold
    
    def _check_exit(self, center):
        """Check if track is near exit zone"""
        # Check if near exit line
        dist = abs(
            (self.exit_line[1][1] - self.exit_line[0][1]) * center[0] -
            (self.exit_line[1][0] - self.exit_line[0][0]) * center[1] +
            self.exit_line[1][0] * self.exit_line[0][1] -
            self.exit_line[1][1] * self.exit_line[0][0]
        ) / np.sqrt(
            (self.exit_line[1][1] - self.exit_line[0][1])**2 +
            (self.exit_line[1][0] - self.exit_line[0][0])**2
        )
        
        if dist < 50:  # Exit zone threshold
            return center
        return None
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity"""
        return np.dot(feat1, feat2) / \
               (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    
    def _get_center(self, bbox):
        """Get bbox center"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _crop_bbox(self, frame, bbox):
        """Crop bbox from frame"""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]
    
    def _get_results(self):
        """Get current results"""
        confirmed_tracks = [
            {
                'track_id': tid,
                'bbox': track.bbox,
                'center': track.center
            }
            for tid, track in self.tracks.items()
            if track.state == 'confirmed'
        ]
        
        avg_fps = 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'tracks': confirmed_tracks,
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'current_inside': len(self.active_people),
            'fps': avg_fps,
            'total_tracks': len(self.tracks)
        }
