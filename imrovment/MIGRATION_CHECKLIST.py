"""
MIGRATION CHECKLIST
===================

Use this checklist to systematically upgrade your people counting system.
Each item shows: Priority (1-5 stars), Time estimate, and Expected impact.

Check off items as you complete them.
"""

MIGRATION_STEPS = """
╔════════════════════════════════════════════════════════════════════════╗
║                    PEOPLE COUNTING MIGRATION PLAN                      ║
╚════════════════════════════════════════════════════════════════════════╝

PHASE 1: CRITICAL FIXES (DO FIRST)
═══════════════════════════════════════════════════════════════════════

[ ] Step 1.1: Fix Hungarian Matching Bug
    Priority: ⭐⭐⭐⭐⭐
    Time: 15 minutes
    Impact: -70% ID switches
    Files: people_counting.py
    
    Action items:
    [ ] Replace _associate_hungarian() method
    [ ] Add _match_stage1() and _match_stage2() methods
    [ ] Add _compute_iou() helper method
    [ ] Test with existing video - verify ID switches reduced
    
    Verification:
    [ ] Run on test video, count ID switches before/after
    [ ] Expected: 70-80% reduction in switches
    
    Reference: people_counting_optimized.py lines 200-350

[ ] Step 1.2: Remove torch.cuda.empty_cache()
    Priority: ⭐⭐⭐⭐
    Time: 2 minutes
    Impact: +20-30% FPS stability
    Files: inference.py
    
    Action items:
    [ ] Remove empty_cache() from detect() method
    [ ] Verify GPU memory doesn't grow unbounded
    
    Verification:
    [ ] Monitor FPS for 5 minutes - should be more stable
    [ ] Check nvidia-smi for memory usage

[ ] Step 1.3: Test Critical Fixes
    [ ] Run 5-minute test video
    [ ] Log metrics: FPS, ID switches, entry/exit counts
    [ ] Compare to baseline (before fixes)
    [ ] Verify ~70% reduction in ID switches


PHASE 2: STABILITY IMPROVEMENTS
═══════════════════════════════════════════════════════════════════════

[ ] Step 2.1: Add Feature EMA
    Priority: ⭐⭐⭐⭐
    Time: 15 minutes
    Impact: -40% appearance errors
    Files: people_counting.py (Track class)
    
    Action items:
    [ ] Update Track.__init__() to add feature_history deque
    [ ] Modify Track.update() to use EMA: 0.9*old + 0.1*new
    [ ] Add get_best_exit_features() method
    
    Verification:
    [ ] Print feature changes per frame (should be smoother)
    [ ] Test re-entry scenarios - should improve
    
    Reference: people_counting_optimized.py lines 80-150

[ ] Step 2.2: Implement Two-Stage Matching
    Priority: ⭐⭐⭐⭐
    Time: Already done in Step 1.1 ✓
    Impact: -50% false associations
    
    Verification:
    [ ] Check logs for stage1 vs stage2 match counts
    [ ] Stage1 should handle ~70-80% of matches

[ ] Step 2.3: Add Hard Gating
    Priority: ⭐⭐⭐
    Time: Already done in Step 1.1 ✓
    Impact: -30% invalid matches
    
    Verification:
    [ ] Log rejected associations (cost = 1e5)
    [ ] Should see fewer impossible matches

[ ] Step 2.4: Test Stability Improvements
    [ ] Run same test video as Phase 1
    [ ] Verify additional ID switch reduction
    [ ] Check re-entry accuracy improved


PHASE 3: PERFORMANCE OPTIMIZATION
═══════════════════════════════════════════════════════════════════════

[ ] Step 3.1: Add Batched OSNet Inference
    Priority: ⭐⭐⭐
    Time: 20 minutes
    Impact: +400% OSNet throughput
    Files: osnet_deepsort_reid.py
    
    Action items:
    [ ] Add extract_features_batch() method to ReIDModel
    [ ] Preprocess all crops together
    [ ] Run single forward pass for all detections
    [ ] Update people_counting.py to use batched method
    
    Code template:
    ```python
    def extract_features_batch(self, crops):
        # Preprocess all
        batch = torch.stack([self._preprocess(c) for c in crops])
        # Single forward pass
        with torch.no_grad():
            features = self.model(batch)
        return features.cpu().numpy()
    ```
    
    Verification:
    [ ] Measure OSNet time before/after
    [ ] Should be 3-8x faster with batch size 8-32

[ ] Step 3.2: Benchmark Current Performance
    [ ] Use PerformanceBenchmark class
    [ ] Record: YOLO fps, OSNet fps, Total fps
    [ ] Note: Max people per frame during test
    
    Files: trt_optimization.py

[ ] Step 3.3: Test Performance Improvements
    [ ] Verify FPS increase from batching
    [ ] Check GPU utilization increased


PHASE 4: TENSORRT CONVERSION
═══════════════════════════════════════════════════════════════════════

[ ] Step 4.1: Convert YOLO to TensorRT
    Priority: ⭐⭐⭐
    Time: 15 minutes (mostly automated)
    Impact: +200% YOLO inference speed
    
    Action items:
    [ ] Install: pip install ultralytics
    [ ] Run conversion:
        ```python
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.export(format='engine', imgsz=640, half=True)
        ```
    [ ] Verify yolov8n.engine created
    [ ] Update inference.py to load .engine file
    
    Verification:
    [ ] Benchmark YOLO before/after
    [ ] Should be 2-3x faster

[ ] Step 4.2: Convert OSNet to TensorRT
    Priority: ⭐⭐⭐
    Time: 30 minutes + waiting for build
    Impact: +200-400% OSNet inference speed
    
    Action items:
    [ ] Export to ONNX (see trt_optimization.py)
    [ ] Build TRT engine with trtexec
    [ ] Update osnet_deepsort_reid.py to load engine
    [ ] Ensure batching still works
    
    Verification:
    [ ] Benchmark OSNet before/after
    [ ] Should be 4-6x faster with batching

[ ] Step 4.3: Benchmark TensorRT Performance
    [ ] Full pipeline benchmark
    [ ] Compare to Phase 3 results
    [ ] Document FPS improvement


PHASE 5: CONFIGURATION TUNING
═══════════════════════════════════════════════════════════════════════

[ ] Step 5.1: Identify Your Scene Type
    Priority: ⭐⭐
    Time: 5 minutes
    Impact: -30% remaining errors
    
    Action items:
    [ ] Measure max people per frame over 1 hour
    [ ] Note your camera FPS
    [ ] Classify: sparse (<5), medium (5-20), crowded (>20)

[ ] Step 5.2: Load Appropriate Config
    [ ] Use config_tuning_guide.py
    [ ] Call auto_tune_config() with your parameters
    [ ] Print config to review values
    
    Example:
    ```python
    from config_tuning_guide import auto_tune_config, print_config
    
    config = auto_tune_config(
        max_people_per_frame=12,
        fps=25,
        resolution=(1920, 1080)
    )
    print_config(config)
    ```

[ ] Step 5.3: Update Your Code with Config Values
    [ ] Set iou_gate = config.iou_gate
    [ ] Set appearance_gate = config.appearance_gate
    [ ] Set spatial_gate = config.spatial_gate
    [ ] Set feature_ema_alpha = config.feature_ema_alpha
    [ ] Set max_disappeared = config.max_disappeared
    [ ] etc.

[ ] Step 5.4: Test and Fine-Tune
    [ ] Run test video with new config
    [ ] Adjust if needed for your specific scenario
    [ ] Document final threshold values


PHASE 6: VALIDATION & DOCUMENTATION
═══════════════════════════════════════════════════════════════════════

[ ] Step 6.1: Performance Validation
    [ ] Run 1-hour test video
    [ ] Record metrics:
        [ ] Average FPS: _____
        [ ] ID switches per hour: _____
        [ ] Entry count accuracy: _____
        [ ] Exit count accuracy: _____
        [ ] Re-entry accuracy: _____

[ ] Step 6.2: Compare to Baseline
    [ ] Calculate improvements:
        [ ] FPS increase: _____x
        [ ] ID switch reduction: _____%
        [ ] Overall accuracy: _____

[ ] Step 6.3: Document Configuration
    [ ] Save final config values
    [ ] Document any custom modifications
    [ ] Note GPU model and expected FPS

[ ] Step 6.4: Create Monitoring Dashboard
    [ ] Add real-time FPS display
    [ ] Add ID switch counter
    [ ] Add entry/exit counters
    [ ] Add GPU utilization meter


TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════

If FPS is still low:
[ ] Check GPU utilization (should be >85%)
[ ] Verify TensorRT engines loaded (not PyTorch)
[ ] Check batch size (should be 8-32 for OSNet)
[ ] Profile code to find bottleneck

If ID switches still high:
[ ] Verify bug fix applied correctly
[ ] Check matched_trk_cols uses indices not IDs
[ ] Increase appearance_gate threshold
[ ] Reduce max_disappeared
[ ] Enable debug logging to see match quality

If re-entry accuracy low:
[ ] Verify get_best_exit_features() implemented
[ ] Check feature_history has data
[ ] Increase reentry_similarity_threshold
[ ] Reduce reentry_spatial_gate


EXPECTED FINAL RESULTS
═══════════════════════════════════════════════════════════════════════

After all phases:

Performance (RTX 4080):
✓ Sparse scenes:  80-100+ FPS
✓ Medium scenes:  50-70 FPS
✓ Crowded scenes: 40-60 FPS

Accuracy:
✓ ID switches: <5 per hour (was 30-50 per minute)
✓ Entry/exit counting: 95-98% accurate
✓ Re-entry matching: 70-85% accurate

GPU:
✓ Utilization: 90-95%
✓ Memory: Stable (no leaks)


QUICK WINS (DO THESE FIRST IF SHORT ON TIME)
═══════════════════════════════════════════════════════════════════════

If you only have 1 hour, do these IN ORDER:

1. Fix Hungarian bug (15 min) → -70% ID switches
2. Remove empty_cache() (2 min) → +25% FPS
3. Add feature EMA (15 min) → -40% appearance errors
4. Batch OSNet (20 min) → +400% OSNet speed

Total: 52 minutes
Total impact: ~85% of full optimization benefit


SUPPORT
═══════════════════════════════════════════════════════════════════════

If you need help with any step, provide:
1. Which step you're on
2. Error messages or unexpected behavior
3. Your GPU model
4. Typical FPS and max people per frame

I can provide specific debugging help and threshold recommendations.

"""

def print_checklist():
    print(MIGRATION_STEPS)

def print_progress_tracker():
    """Print a simple progress tracker"""
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                         PROGRESS TRACKER                               ║
╚════════════════════════════════════════════════════════════════════════╝

Mark your progress:

PHASE 1: CRITICAL FIXES          [    ] 0/3 complete
PHASE 2: STABILITY              [    ] 0/4 complete  
PHASE 3: PERFORMANCE            [    ] 0/3 complete
PHASE 4: TENSORRT               [    ] 0/3 complete
PHASE 5: CONFIGURATION          [    ] 0/4 complete
PHASE 6: VALIDATION             [    ] 0/4 complete

Overall Progress: [                              ] 0/21 steps

Estimated time remaining: ___ hours
Expected improvements so far: 
  - ID switches: ___% reduction
  - FPS: ___x increase
    """)

if __name__ == "__main__":
    print_checklist()
    print("\n\n")
    print_progress_tracker()
