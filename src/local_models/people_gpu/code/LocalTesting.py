"""
Enhanced Testing Script for People Counting System
- return_annotated=False → Save all frames to output.json
- return_annotated=True → Show inference/annotated frames in real-time
"""

import os
import sys
import json
import base64
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Import inference functions
from inference import model_fn, input_fn, predict_fn, output_fn


class EnhancedTester:
    def __init__(self, model_dir, output_dir="test_outputs"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model_loaded = False
        self.frame_results = []

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load the model"""
        print(f"Loading model from: {self.model_dir}")
        result = model_fn(self.model_dir)

        if result.get("status") in ["loaded", "already_loaded"]:
            self.model_loaded = True
            print("✓ Model loaded successfully")
            print(f"  Device: {result.get('device', 'unknown')}")
            print(f"  GPU Available: {result.get('gpu_available', False)}")
            return True
        else:
            print(f"✗ Failed to load model: {result.get('error')}")
            return False

    def process_frame(self, frame, camid=1, threshold=10, alert_rate=60,
                     return_annotated=False, confidence_threshold=0.35):
        """Process a single frame"""
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        encoding = base64.b64encode(buffer).decode('utf-8')

        # Create request
        request_data = {
            "camid": camid,
            "org_id": 1,
            "userid": 1,
            "encoding": encoding,
            "threshold": threshold,
            "alert_rate": alert_rate,
            "return_annotated": return_annotated,
            "confidence_threshold": confidence_threshold
        }

        # Process
        request_body = json.dumps(request_data)
        input_data = input_fn(request_body, content_type="application/json")
        prediction = predict_fn(input_data)
        output_json = output_fn(prediction, content_type="application/json")

        return json.loads(output_json)

    def save_results_to_json(self, filename="output.json"):
        """Save all frame results to JSON file"""
        output_path = os.path.join(self.output_dir, filename)

        # Create summary
        summary = {
            "test_info": {
                "total_frames": len(self.frame_results),
                "test_date": datetime.now().isoformat(),
                "model_dir": self.model_dir
            },
            "frames": self.frame_results
        }

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")
        print(f"  Total frames: {len(self.frame_results)}")
        return output_path

    def show_results(self, result):
        """Print frame results to console"""
        print(f"  People: {result.get('Total_people_detected', 0):2d} | "
              f"Occupancy: {result.get('Occupancy_percentage', 0):5.1f}% | "
              f"Status: {result.get('Status', 'Unknown'):15s} | "
              f"Alert: {result.get('is_alert_triggered', False)}")

    def test_video(self, video_path, camid=1, threshold=10, alert_rate=60,
                   return_annotated=False, confidence_threshold=0.35,
                   max_frames=None):
        """
        Test with video file

        Args:
            video_path: Path to video file
            camid: Camera ID
            threshold: Maximum capacity
            alert_rate: Alert percentage (60 = alert at 60%)
            return_annotated: False = save to JSON, True = show inference
            confidence_threshold: YOLO confidence threshold
            max_frames: Maximum frames to process (None = all)
        """
        print(f"\n{'='*70}")
        print(f"Processing video: {video_path}")
        print(f"Mode: {'INFERENCE (Show Frames)' if return_annotated else 'OUTPUT (Save JSON)'}")
        print(f"Config: threshold={threshold}, alert_rate={alert_rate}%, conf={confidence_threshold}")
        print(f"{'='*70}\n")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Failed to open video")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Info: {total_frames} frames @ {fps} FPS\n")

        frame_count = 0
        self.frame_results = []  # Reset results

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Check max frames limit
            if max_frames and frame_count > max_frames:
                print(f"\n⚠ Reached max frames limit ({max_frames})")
                break

            print(f"Frame {frame_count:4d}/{total_frames:4d} ", end="")

            # Process frame
            result = self.process_frame(
                frame,
                camid=camid,
                threshold=threshold,
                alert_rate=alert_rate,
                return_annotated=return_annotated,
                confidence_threshold=confidence_threshold
            )

            # Show results
            self.show_results(result)

            if return_annotated:
                # MODE 1: Show inference (annotated frames)
                if result.get('annotated_frame'):
                    img_data = base64.b64decode(result['annotated_frame'])
                    nparr = np.frombuffer(img_data, np.uint8)
                    annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Add frame counter on image
                    #cv2.putText(annotated, f"Frame: {frame_count}", (10, annotated.shape[0] - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow('People Counter (Press Q to quit, Space to pause)', annotated)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("\n⚠ User quit")
                        break
                    elif key == ord(' '):
                        print("⏸ Paused (press any key to continue)")
                        cv2.waitKey(0)
            else:
                # MODE 2: Save to JSON (remove annotated_frame to save space)
                result_copy = result.copy()
                if 'annotated_frame' in result_copy:
                    del result_copy['annotated_frame']
                self.frame_results.append(result_copy)

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*70}")
        print(f"✓ Processed {frame_count} frames")
        print(f"{'='*70}\n")

        # Save results if in JSON mode
        if not return_annotated and self.frame_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.json"
            self.save_results_to_json(filename)
            self.generate_summary_report(filename.replace('.json', '_summary.txt'))

    def test_image(self, image_path, camid=1, threshold=10, alert_rate=60,
                   return_annotated=True, confidence_threshold=0.35):
        """Test with single image"""
        print(f"\n{'='*70}")
        print(f"Processing image: {image_path}")
        print(f"{'='*70}\n")

        frame = cv2.imread(image_path)
        if frame is None:
            print("✗ Failed to read image")
            return

        result = self.process_frame(
            frame,
            camid=camid,
            threshold=threshold,
            alert_rate=alert_rate,
            return_annotated=return_annotated,
            confidence_threshold=confidence_threshold
        )

        # Print detailed results
        print(f"\nResults:")
        print(f"  Frame ID: {result.get('Frame_Id')}")
        print(f"  Timestamp: {result.get('Time_stamp')}")
        print(f"  People Detected: {result.get('Total_people_detected')}")
        print(f"  People IDs: {result.get('People_ids')}")
        print(f"  Occupancy: {result.get('Occupancy_percentage')}%")
        print(f"  Status: {result.get('Status')}")
        print(f"  Alert Triggered: {result.get('is_alert_triggered')}")
        print(f"  Total Entries: {result.get('Total_entries')}")
        print(f"  Total Exits: {result.get('Total_exits')}")
        print(f"  Net Count: {result.get('Net_count')}")

        # Show annotated frame
        if return_annotated and result.get('annotated_frame'):
            img_data = base64.b64decode(result['annotated_frame'])
            nparr = np.frombuffer(img_data, np.uint8)
            annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow('People Counter (Press any key to close)', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save result
        if not return_annotated:
            result_copy = result.copy()
            if 'annotated_frame' in result_copy:
                del result_copy['annotated_frame']

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_output_{timestamp}.json"
            output_path = os.path.join(self.output_dir, filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, indent=2, ensure_ascii=False)

            print(f"\n✓ Result saved to: {output_path}")

    def generate_summary_report(self, filename):
        """Generate a text summary report"""
        if not self.frame_results:
            return

        output_path = os.path.join(self.output_dir, filename)

        # Calculate statistics
        total_frames = len(self.frame_results)
        people_counts = [r.get('Total_people_detected', 0) for r in self.frame_results]
        occupancies = [r.get('Occupancy_percentage', 0) for r in self.frame_results]
        alerts = [r.get('is_alert_triggered', False) for r in self.frame_results]

        avg_people = sum(people_counts) / total_frames if total_frames > 0 else 0
        max_people = max(people_counts) if people_counts else 0
        min_people = min(people_counts) if people_counts else 0
        avg_occupancy = sum(occupancies) / total_frames if total_frames > 0 else 0
        alert_count = sum(alerts)

        # Get final stats
        final_frame = self.frame_results[-1] if self.frame_results else {}

        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PEOPLE COUNTING TEST SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Processed: {total_frames}\n\n")

            f.write("DETECTION STATISTICS:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Average People per Frame: {avg_people:.2f}\n")
            f.write(f"  Maximum People: {max_people}\n")
            f.write(f"  Minimum People: {min_people}\n")
            f.write(f"  Average Occupancy: {avg_occupancy:.1f}%\n")
            f.write(f"  Alert Triggers: {alert_count}\n\n")

            f.write("FINAL COUNTS:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Total Entries: {final_frame.get('Total_entries', 0)}\n")
            f.write(f"  Total Exits: {final_frame.get('Total_exits', 0)}\n")
            f.write(f"  Net Count: {final_frame.get('Net_count', 0)}\n")
            f.write(f"  Current Occupancy: {final_frame.get('Current_occupancy', 0)}\n\n")

            f.write("FRAME-BY-FRAME SUMMARY:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Frame':<8} {'People':<8} {'Occupancy':<12} {'Status':<20} {'Alert':<8}\n")
            f.write("-"*70 + "\n")

            for i, result in enumerate(self.frame_results, 1):
                people = result.get('Total_people_detected', 0)
                occ = result.get('Occupancy_percentage', 0)
                status = result.get('Status', '')
                alert = 'YES' if result.get('is_alert_triggered', False) else 'NO'

                f.write(f"{i:<8} {people:<8} {occ:<12.1f} {status:<20} {alert:<8}\n")

        print(f"✓ Summary report saved to: {output_path}")


# Quick usage examples
if __name__ == "__main__":
    # Configuration
    MODEL_DIR = r"E:\UTC project\utc\cctv\CCTV_Project\Project_CCTV_arun\21Nov2025-People"
    VIDEO_PATH = r"E:\UTC project\utc\cctv\CCTV_Project\Project_CCTV_arun\21Nov2025-People\Vid.mp4"
    #VIDEO_PATH = 0
    # Create tester
    tester = EnhancedTester(MODEL_DIR, output_dir="test_outputs")

    # Load model
    if not tester.load_model():
        sys.exit(1)

    print("\n" + "="*70)
    print("TESTING OPTIONS:")
    print("="*70)
    print("1. return_annotated=False → Save all frames to output.json")
    print("2. return_annotated=True  → Show inference (annotated frames)")
    print("="*70 + "\n")

    # Choose mode
    mode = input("Enter mode (1=JSON output, 2=Show inference): ").strip()

    if mode == "1":
        # MODE 1: Save to JSON (no visualization)
        print("\n📄 JSON OUTPUT MODE - Saving all frames to output.json")
        tester.test_video(
            VIDEO_PATH,
            camid=2,
            threshold=10,
            alert_rate=70,
            return_annotated=False,  # Save to JSON
            confidence_threshold=0.35,
            max_frames=None  # Process all frames
        )

    elif mode == "2":
        # MODE 2: Show inference (with visualization)
        print("\n🎥 INFERENCE MODE - Showing annotated frames")
        tester.test_video(
            VIDEO_PATH,
            camid=1,
            threshold=10,
            alert_rate=60,
            return_annotated=True,  # Show frames
            confidence_threshold=0.40,
            max_frames=None  # Process all frames
        )

    else:
        print("Invalid mode selected!")
        sys.exit(1)

    print("\n✅ Testing completed!")