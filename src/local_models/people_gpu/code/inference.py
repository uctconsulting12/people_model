"""
People Counting Inference Handler - Production Version
File: inference.py

================================================================================
INPUT PAYLOAD SPECIFICATION
================================================================================

{
  "camid": 1,                          // integer, required - Camera identifier
  "org_id": 100,                       // integer, required - Organization ID
  "userid": 42,                        // integer, required - User ID
  "encoding": "base64...",             // string, required - Base64-encoded image (JPEG/PNG)
  "threshold": 10,                     // integer, required - Maximum occupancy limit (dynamic: 5, 10, 20, 50, 100, etc.)
  "alert_rate": 80,                    // integer, required - Critical alert percentage (0-100)
  "return_annotated": true,            // boolean, required - Return annotated frame?
  "confidence_threshold": 0.35         // float, optional - YOLO confidence (default: 0.35, range: 0.01-0.99)
}

NOTE: Warning alerts are FIXED at 65% occupancy (not user-controlled).

================================================================================
OUTPUT PAYLOAD SPECIFICATION
================================================================================

SUCCESS RESPONSE:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789123",
  "Time_stamp": "2024-12-17T15:39:49.123456Z",
  "Frame_Count": 42,
  "Total_people_detected": 3,
  "Current_occupancy": 3,
  "People_ids": [5, 12, 23],
  "Entry_time": ["2024-12-17T15:39:45.000Z", ...],
  "Exit_time": ["2024-12-17T15:39:47.800Z"],
  "exitid": [7],
  "People_dwell_time": [4.123, 2.623, 0.923],
  "Confidence_scores": [0.87, 0.91, 0.78],
  "Bounding_boxes": [[100, 200, 250, 500], ...],
  "x": [100, 350, 520],
  "y": [200, 180, 210],
  "w": [150, 130, 130],
  "h": [300, 340, 280],
  "accuracy": [0.870, 0.910, 0.780],
  "Total_entries": 15,
  "Total_exits": 12,
  "Net_count": 3,
  "Occupancy_percentage": 30.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 2.556,
  "Max_occupancy": 10,
  "Status": "Warning",
  "is_alert_triggered": true,
  "Processing_Status": 1,
  "annotated_frame": "base64...",
  "processing_time_ms": 85.43
}

ERROR RESPONSE:
{
  "camid": 1,
  "Frame_Id": "ERROR_1734456789123",
  "Status": "Error",
  "Processing_Status": 0,
  "error_message": "Model not initialized",
  ... (all other fields set to 0/empty)
}

================================================================================
DIFFERENT SCENARIOS
================================================================================

SCENARIO 1: Normal Operation (No Alert)
---------------------------------------
Input:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "encoding": "base64_encoded_image...",
  "threshold": 10,
  "alert_rate": 80,
  "return_annotated": true,
  "confidence_threshold": 0.35
}

Output:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789001",
  "Time_stamp": "2024-12-17T10:00:00.000000Z",
  "Frame_Count": 1,
  "Total_people_detected": 3,
  "Current_occupancy": 3,
  "People_ids": [1, 2, 3],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T09:59:58.000Z"
  ],
  "Exit_time": [],
  "exitid": [],
  "People_dwell_time": [5.0, 3.0, 2.0],
  "Confidence_scores": [0.89, 0.92, 0.85],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440]
  ],
  "x": [175, 365, 565],
  "y": [300, 300, 300],
  "w": [150, 130, 130],
  "h": [300, 320, 280],
  "accuracy": [0.890, 0.920, 0.850],
  "Total_entries": 3,
  "Total_exits": 0,
  "Net_count": 3,
  "Occupancy_percentage": 30.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 3.33,
  "Max_occupancy": 10,
  "Status": "",
  "is_alert_triggered": false,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 87.5
}

SCENARIO 2: Warning Level (64-79% occupancy)
--------------------------------------------
Input:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "encoding": "base64_encoded_image...",
  "threshold": 10,
  "alert_rate": 80,
  "return_annotated": true
}

Calculation: 7 people / 10 threshold * 100 = 70.0%
Check: 70 >= 80? NO, 70 >= 64? YES → "Warning"

Output:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789002",
  "Time_stamp": "2024-12-17T10:05:00.000000Z",
  "Frame_Count": 42,
  "Total_people_detected": 7,
  "Current_occupancy": 7,
  "People_ids": [1, 2, 3, 4, 5, 6, 7],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T10:00:10.000Z",
    "2024-12-17T10:01:20.000Z",
    "2024-12-17T10:02:30.000Z",
    "2024-12-17T10:03:15.000Z",
    "2024-12-17T10:04:50.000Z"
  ],
  "Exit_time": [],
  "exitid": [],
  "People_dwell_time": [305.0, 303.0, 290.0, 220.0, 150.0, 105.0, 10.0],
  "Confidence_scores": [0.89, 0.92, 0.85, 0.88, 0.91, 0.87, 0.84],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440],
    [150, 200, 280, 500],
    [350, 180, 480, 480],
    [550, 170, 680, 470],
    [200, 190, 330, 490]
  ],
  "x": [175, 365, 565, 215, 415, 615, 265],
  "y": [300, 300, 300, 350, 330, 320, 340],
  "w": [150, 130, 130, 130, 130, 130, 130],
  "h": [300, 320, 280, 300, 300, 300, 300],
  "accuracy": [0.890, 0.920, 0.850, 0.880, 0.910, 0.870, 0.840],
  "Total_entries": 7,
  "Total_exits": 0,
  "Net_count": 7,
  "Occupancy_percentage": 70.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 197.57,
  "Max_occupancy": 10,
  "Status": "Warning",
  "is_alert_triggered": true,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 92.3
}

SCENARIO 3: Alert Level (≥80% occupancy)
----------------------------------------
Input:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "encoding": "base64_encoded_image...",
  "threshold": 10,
  "alert_rate": 80,
  "return_annotated": true
}

Calculation: 8 people / 10 threshold * 100 = 80.0%
Check: 80 >= 80? YES → "Alert"

Output:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789003",
  "Time_stamp": "2024-12-17T10:10:00.000000Z",
  "Frame_Count": 50,
  "Total_people_detected": 8,
  "Current_occupancy": 8,
  "People_ids": [1, 2, 3, 4, 5, 6, 7, 8],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T10:00:10.000Z",
    "2024-12-17T10:01:20.000Z",
    "2024-12-17T10:02:30.000Z",
    "2024-12-17T10:03:15.000Z",
    "2024-12-17T10:04:50.000Z",
    "2024-12-17T10:09:55.000Z"
  ],
  "Exit_time": [],
  "exitid": [],
  "People_dwell_time": [605.0, 603.0, 590.0, 520.0, 450.0, 405.0, 310.0, 5.0],
  "Confidence_scores": [0.89, 0.92, 0.85, 0.88, 0.91, 0.87, 0.84, 0.90],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440],
    [150, 200, 280, 500],
    [350, 180, 480, 480],
    [550, 170, 680, 470],
    [200, 190, 330, 490],
    [400, 200, 530, 500]
  ],
  "x": [175, 365, 565, 215, 415, 615, 265, 465],
  "y": [300, 300, 300, 350, 330, 320, 340, 350],
  "w": [150, 130, 130, 130, 130, 130, 130, 130],
  "h": [300, 320, 280, 300, 300, 300, 300, 300],
  "accuracy": [0.890, 0.920, 0.850, 0.880, 0.910, 0.870, 0.840, 0.900],
  "Total_entries": 8,
  "Total_exits": 0,
  "Net_count": 8,
  "Occupancy_percentage": 80.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 436.0,
  "Max_occupancy": 10,
  "Status": "Alert",
  "is_alert_triggered": true,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 95.7
}

SCENARIO 4: Alert Suppressed (State-Based Debouncing)
-----------------------------------------------------
Frame 1 Output (Alert First Triggered):
{
  "Frame_Id": "FR_1_1734456789003",
  "Frame_Count": 50,
  "Total_people_detected": 8,
  "Occupancy_percentage": 80.0,
  "Status": "Alert",
  "is_alert_triggered": true,
  "Processing_Status": 1
}

Frame 2 Output (Same Condition - Alert Suppressed):
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789004",
  "Time_stamp": "2024-12-17T10:10:01.000000Z",
  "Frame_Count": 51,
  "Total_people_detected": 8,
  "Current_occupancy": 8,
  "People_ids": [1, 2, 3, 4, 5, 6, 7, 8],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T10:00:10.000Z",
    "2024-12-17T10:01:20.000Z",
    "2024-12-17T10:02:30.000Z",
    "2024-12-17T10:03:15.000Z",
    "2024-12-17T10:04:50.000Z",
    "2024-12-17T10:09:55.000Z"
  ],
  "Exit_time": [],
  "exitid": [],
  "People_dwell_time": [606.0, 604.0, 591.0, 521.0, 451.0, 406.0, 311.0, 6.0],
  "Confidence_scores": [0.89, 0.92, 0.85, 0.88, 0.91, 0.87, 0.84, 0.90],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440],
    [150, 200, 280, 500],
    [350, 180, 480, 480],
    [550, 170, 680, 470],
    [200, 190, 330, 490],
    [400, 200, 530, 500]
  ],
  "x": [175, 365, 565, 215, 415, 615, 265, 465],
  "y": [300, 300, 300, 350, 330, 320, 340, 350],
  "w": [150, 130, 130, 130, 130, 130, 130, 130],
  "h": [300, 320, 280, 300, 300, 300, 300, 300],
  "accuracy": [0.890, 0.920, 0.850, 0.880, 0.910, 0.870, 0.840, 0.900],
  "Total_entries": 8,
  "Total_exits": 0,
  "Net_count": 8,
  "Occupancy_percentage": 80.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 437.0,
  "Max_occupancy": 10,
  "Status": "",
  "is_alert_triggered": false,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 89.2
}

Note: Status is empty ("") and is_alert_triggered is false because alert is
      suppressed (alert already active from Frame 1)

SCENARIO 5: Alert Cleared and Re-triggered
------------------------------------------
Frame 1 (Alert Triggered):
{
  "Frame_Count": 50,
  "Total_people_detected": 8,
  "Occupancy_percentage": 80.0,
  "Status": "Alert",
  "is_alert_triggered": true
}

Frame 2-10 (Alert Suppressed - Same 8 people):
{
  "Frame_Count": 51-59,
  "Total_people_detected": 8,
  "Occupancy_percentage": 80.0,
  "Status": "",
  "is_alert_triggered": false
}

Frame 11 (Occupancy Drops - Alert Cleared):
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789014",
  "Time_stamp": "2024-12-17T10:10:11.000000Z",
  "Frame_Count": 60,
  "Total_people_detected": 6,
  "Current_occupancy": 6,
  "People_ids": [1, 2, 3, 4, 5, 6],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T10:00:10.000Z",
    "2024-12-17T10:01:20.000Z",
    "2024-12-17T10:02:30.000Z",
    "2024-12-17T10:03:15.000Z"
  ],
  "Exit_time": [
    "2024-12-17T10:10:10.000Z",
    "2024-12-17T10:10:10.500Z"
  ],
  "exitid": [7, 8],
  "People_dwell_time": [616.0, 614.0, 601.0, 531.0, 461.0, 416.0],
  "Confidence_scores": [0.89, 0.92, 0.85, 0.88, 0.91, 0.87],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440],
    [150, 200, 280, 500],
    [350, 180, 480, 480],
    [550, 170, 680, 470]
  ],
  "x": [175, 365, 565, 215, 415, 615],
  "y": [300, 300, 300, 350, 330, 320],
  "w": [150, 130, 130, 130, 130, 130],
  "h": [300, 320, 280, 300, 300, 300],
  "accuracy": [0.890, 0.920, 0.850, 0.880, 0.910, 0.870],
  "Total_entries": 8,
  "Total_exits": 2,
  "Net_count": 6,
  "Occupancy_percentage": 60.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 539.83,
  "Max_occupancy": 10,
  "Status": "",
  "is_alert_triggered": false,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 88.1
}

Note: Occupancy 60% < 64% (reset threshold), so alert state cleared

Frame 12 (New Person Enters - Alert Re-triggered):
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789015",
  "Time_stamp": "2024-12-17T10:10:12.000000Z",
  "Frame_Count": 61,
  "Total_people_detected": 8,
  "Current_occupancy": 8,
  "People_ids": [1, 2, 3, 4, 5, 6, 9, 10],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T10:00:10.000Z",
    "2024-12-17T10:01:20.000Z",
    "2024-12-17T10:02:30.000Z",
    "2024-12-17T10:03:15.000Z",
    "2024-12-17T10:10:11.500Z",
    "2024-12-17T10:10:11.800Z"
  ],
  "Exit_time": [
    "2024-12-17T10:10:10.000Z",
    "2024-12-17T10:10:10.500Z"
  ],
  "exitid": [7, 8],
  "People_dwell_time": [617.0, 615.0, 602.0, 532.0, 462.0, 417.0, 0.5, 0.2],
  "Confidence_scores": [0.89, 0.92, 0.85, 0.88, 0.91, 0.87, 0.86, 0.88],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440],
    [150, 200, 280, 500],
    [350, 180, 480, 480],
    [550, 170, 680, 470],
    [250, 195, 380, 495],
    [450, 205, 580, 505]
  ],
  "x": [175, 365, 565, 215, 415, 615, 315, 515],
  "y": [300, 300, 300, 350, 330, 320, 345, 355],
  "w": [150, 130, 130, 130, 130, 130, 130, 130],
  "h": [300, 320, 280, 300, 300, 300, 300, 300],
  "accuracy": [0.890, 0.920, 0.850, 0.880, 0.910, 0.870, 0.860, 0.880],
  "Total_entries": 10,
  "Total_exits": 2,
  "Net_count": 8,
  "Occupancy_percentage": 80.0,
  "Over_capacity_count": 0,
  "Average_dwell_time": 405.59,
  "Max_occupancy": 10,
  "Status": "Alert",
  "is_alert_triggered": true,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 91.4
}

Note: NEW alert triggered because alert state was cleared in Frame 11

SCENARIO 6: Over Capacity
-------------------------
Input:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "encoding": "base64_encoded_image...",
  "threshold": 10,
  "alert_rate": 80,
  "return_annotated": true
}

Calculation: 15 people / 10 threshold * 100 = 150.0%

Output:
{
  "camid": 1,
  "org_id": 100,
  "userid": 42,
  "Frame_Id": "FR_1_1734456789020",
  "Time_stamp": "2024-12-17T10:20:00.000000Z",
  "Frame_Count": 100,
  "Total_people_detected": 15,
  "Current_occupancy": 15,
  "People_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "Entry_time": [
    "2024-12-17T09:59:55.000Z",
    "2024-12-17T09:59:57.000Z",
    "2024-12-17T10:00:10.000Z",
    "2024-12-17T10:01:20.000Z",
    "2024-12-17T10:02:30.000Z",
    "2024-12-17T10:03:15.000Z",
    "2024-12-17T10:04:50.000Z",
    "2024-12-17T10:09:55.000Z",
    "2024-12-17T10:12:30.000Z",
    "2024-12-17T10:14:15.000Z",
    "2024-12-17T10:15:45.000Z",
    "2024-12-17T10:17:10.000Z",
    "2024-12-17T10:18:20.000Z",
    "2024-12-17T10:19:00.000Z",
    "2024-12-17T10:19:50.000Z"
  ],
  "Exit_time": [
    "2024-12-17T10:05:30.000Z",
    "2024-12-17T10:08:45.000Z"
  ],
  "exitid": [16, 17],
  "People_dwell_time": [1205.0, 1203.0, 1190.0, 1120.0, 1050.0, 1005.0, 910.0, 605.0, 450.0, 345.0, 255.0, 170.0, 100.0, 60.0, 10.0],
  "Confidence_scores": [0.89, 0.92, 0.85, 0.88, 0.91, 0.87, 0.84, 0.90, 0.86, 0.89, 0.88, 0.85, 0.87, 0.91, 0.83],
  "Bounding_boxes": [
    [100, 150, 250, 450],
    [300, 140, 430, 460],
    [500, 160, 630, 440],
    [150, 200, 280, 500],
    [350, 180, 480, 480],
    [550, 170, 680, 470],
    [200, 190, 330, 490],
    [400, 200, 530, 500],
    [250, 195, 380, 495],
    [450, 205, 580, 505],
    [600, 190, 730, 490],
    [120, 210, 250, 510],
    [320, 205, 450, 505],
    [520, 195, 650, 495],
    [380, 215, 510, 515]
  ],
  "x": [175, 365, 565, 215, 415, 615, 265, 465, 315, 515, 665, 185, 385, 585, 445],
  "y": [300, 300, 300, 350, 330, 320, 340, 350, 345, 355, 340, 360, 355, 345, 365],
  "w": [150, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130],
  "h": [300, 320, 280, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
  "accuracy": [0.890, 0.920, 0.850, 0.880, 0.910, 0.870, 0.840, 0.900, 0.860, 0.890, 0.880, 0.850, 0.870, 0.910, 0.830],
  "Total_entries": 17,
  "Total_exits": 2,
  "Net_count": 15,
  "Occupancy_percentage": 150.0,
  "Over_capacity_count": 5,
  "Average_dwell_time": 645.8,
  "Max_occupancy": 10,
  "Status": "Alert",
  "is_alert_triggered": true,
  "Processing_Status": 1,
  "annotated_frame": "base64_encoded_annotated_image...",
  "processing_time_ms": 102.3
}

Note: Occupancy_percentage is 150% (not clamped at 100%)
      Over_capacity_count is 5 (15 - 10 = 5 people over threshold)

================================================================================
ALERT MANAGEMENT LOGIC (State-Based Debouncing)
================================================================================

PURPOSE: Prevent alert spam while maintaining accurate threat awareness

CALCULATION LOGIC:
------------------
1. occupancy_percentage = (current_people / threshold) * 100
2. Determine actual_status:
   - IF occupancy >= alert_rate          → "Alert"
   - ELIF occupancy >= alert_rate * 0.8  → "Warning"
   - ELSE                                → "" (empty string)

3. reset_threshold = alert_rate * 0.8  (for hysteresis)

STATE MACHINE:
--------------
┌─────────────────┐
│   INITIAL       │  alert_active = False
│   (No Alert)    │
└────────┬────────┘
         │
         │ Condition: actual_status in ["Alert", "Warning"] AND alert_active = False
         │ Action: Send NEW alert, set alert_active = True
         ↓
┌─────────────────┐
│  ALERT ACTIVE   │  alert_active = True
│  (Suppressing)  │◄─────┐
└────────┬────────┘      │
         │               │ Condition: actual_status in ["Alert", "Warning"] AND alert_active = True
         │               │ Action: SUPPRESS alert (no new alert sent)
         ├───────────────┘
         │
         │ Condition: occupancy < reset_threshold AND alert_active = True
         │ Action: CLEAR alert state, set alert_active = False
         ↓
┌─────────────────┐
│   INITIAL       │  Ready for new alert
│   (Cleared)     │
└─────────────────┘

STATE TRANSITIONS:
------------------
1. INITIAL → ALERT ACTIVE
   - Trigger: Problem detected AND no active alert
   - Result: is_alert_triggered = true, Status shows "Alert" or "Warning"

2. ALERT ACTIVE → ALERT ACTIVE
   - Trigger: Problem persists AND alert already active
   - Result: is_alert_triggered = false, Status = "" (suppressed)

3. ALERT ACTIVE → INITIAL
   - Trigger: Occupancy drops below reset_threshold (alert_rate * 0.8)
   - Result: Alert state cleared, ready for new alert

4. INITIAL → INITIAL
   - Trigger: No problem detected
   - Result: is_alert_triggered = false, Status = ""

EXAMPLES:
---------
Example 1: Continuous High Occupancy (5 minutes = ~300 frames)
Frame 1:   8 people → Alert SENT ✓
Frame 2:   8 people → Alert SUPPRESSED
Frame 3:   8 people → Alert SUPPRESSED
...
Frame 300: 8 people → Alert SUPPRESSED
Result: 1 alert total (not 300 alerts!)

Example 2: Alert → Warning → Clear
Frame 1: 9 people (90%) → Alert SENT, alert_active=True
Frame 2: 8 people (80%) → Alert SUPPRESSED (still active)
Frame 3: 7 people (70%) → Alert SUPPRESSED (Warning range, still active)
Frame 4: 6 people (60%) → Alert CLEARED (60 < 64% reset), alert_active=False
Frame 5: 7 people (70%) → NEW Warning SENT, alert_active=True

Example 3: Multiple Cameras (Independent State)
Camera 1 Frame 1: Alert active
Camera 2 Frame 1: No alert
Camera 1 Frame 2: Alert suppressed (independent of Camera 2)
Camera 2 Frame 2: Alert triggered (independent of Camera 1)

HYSTERESIS EXPLANATION:
-----------------------
- Alert triggers at: occupancy >= alert_rate (e.g., 80%)
- Alert clears at: occupancy < alert_rate * 0.8 (e.g., 64%)
- This 16% buffer prevents flickering alerts near threshold
- Example: If hovering at 75%, alert won't flicker on/off

BENEFITS:
---------
✓ No alert spam (one alert per incident)
✓ Hysteresis prevents flickering
✓ Per-camera independent state
✓ Clears automatically when resolved
✓ New alert when problem re-occurs

================================================================================
FIELD TYPES REFERENCE
================================================================================

INPUT TYPES:
  camid: integer
  org_id: integer
  userid: integer
  encoding: string (base64)
  threshold: integer
  alert_rate: integer
  return_annotated: boolean
  confidence_threshold: float (optional)

OUTPUT TYPES:
  camid, org_id, userid: integer
  Frame_Id, Time_stamp: string
  Frame_Count, Total_people_detected, Current_occupancy: integer
  People_ids, exitid: array of integers
  Entry_time, Exit_time: array of strings (ISO 8601)
  People_dwell_time, Confidence_scores, accuracy: array of floats
  Bounding_boxes: array of [4 integers]
  x, y, w, h: array of integers
  Total_entries, Total_exits, Net_count: integer
  Occupancy_percentage, Average_dwell_time: float
  Over_capacity_count, Max_occupancy: integer
  Status: string ("", "Warning", "Alert", "Error")
  is_alert_triggered: boolean
  Processing_Status: integer (0=error, 1=success)
  annotated_frame: string (base64) or null
  processing_time_ms: float
  error_message: string (only in errors)

================================================================================
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core dependencies
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

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
    logger.info("NVIDIA ML available")
except Exception as e:
    pynvml = None
    NVIDIA_ML_AVAILABLE = False
    logger.warning(f"NVIDIA ML not available: {e}")

# Import people counting system
try:
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

# Global variables
yolo_model = None
system_manager = None
model_loaded = False
device = None
_lock = threading.Lock()


def find_model_file(model_dir: str) -> str:
    """Find people8s.pt model file"""
    logger.info(f"Searching for people8s.pt in: {model_dir}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    contents = os.listdir(model_dir)
    logger.info(f"Directory contents: {contents}")

    model_path = os.path.join(model_dir, "people8s.pt")

    if os.path.exists(model_path) and os.path.isfile(model_path):
        logger.info(f"Found required model: people8s.pt")
        return model_path

    pt_files = [f for f in contents if f.endswith('.pt')]
    if pt_files:
        selected = os.path.join(model_dir, pt_files[0])
        logger.warning(f"people8s.pt not found, using fallback: {pt_files[0]}")
        return selected

    raise FileNotFoundError(f"Required model people8s.pt not found in {model_dir}")


def find_osnet_weights(model_dir: str) -> Optional[str]:
    """Find OSNet pretrained weights file"""
    possible_paths = [
        os.path.join(model_dir, "pretrained", "osnet_x0_25_imagenet.pth"),
        os.path.join(model_dir, "osnet_x0_25_imagenet.pth"),
        os.path.join(os.path.dirname(model_dir), "pretrained", "osnet_x0_25_imagenet.pth"),
        os.path.join(current_dir, "pretrained", "osnet_x0_25_imagenet.pth")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✓ Found OSNet weights: {path}")
            return path

    logger.error(f"❌ OSNet weights not found")
    logger.warning("⚠ Will use randomly initialized weights (lower accuracy)")
    return None


def setup_device():
    """Setup device optimized for GPU/CPU"""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, using CPU")
        return "cpu"

    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA devices found: {device_count}")

            device_obj = torch.device("cuda:0")
            torch.cuda.set_device(device_obj)
            torch.cuda.set_per_process_memory_fraction(0.6)
            torch.cuda.empty_cache()

            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")

            return device_obj
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

    except Exception as e:
        logger.error(f"Device setup failed: {e}, falling back to CPU")
        return "cpu"


def get_gpu_stats() -> Dict[str, float]:
    """Get GPU statistics for monitoring"""
    stats = {
        "gpu_utilization_percent": 0.0,
        "gpu_memory_percent": 0.0,
        "gpu_memory_used_mb": 0.0,
        "gpu_temperature_c": 0.0
    }

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return stats

    try:
        if NVIDIA_ML_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu_utilization_percent"] = util.gpu

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["gpu_memory_percent"] = (mem_info.used / mem_info.total) * 100
            stats["gpu_memory_used_mb"] = mem_info.used / 1024 ** 2

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                stats["gpu_temperature_c"] = temp
            except:
                pass
        else:
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            stats["gpu_memory_used_mb"] = allocated
            stats["gpu_memory_percent"] = (allocated / total) * 100

    except Exception as e:
        logger.debug(f"GPU stats failed: {e}")

    return stats


def log_metrics_to_console(metrics: Dict[str, float], camid: int, org_id: int):
    """Log metrics to console"""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("=" * 60)
        logger.info(f"METRICS - Camera: {camid} | Organization: {org_id}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info("-" * 60)

        for name, value in metrics.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                logger.info(f"  {name}: {value}")

        logger.info("=" * 60)

    except Exception as e:
        logger.debug(f"Metric logging failed: {e}")


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load model with comprehensive error handling"""
    global yolo_model, system_manager, model_loaded, device

    with _lock:
        if model_loaded:
            logger.info("Model already loaded")
            return {"status": "already_loaded", "device": str(device)}

        try:
            logger.info("=" * 60)
            logger.info("MODEL LOADING - People Counting with State-Based Alert Debouncing")
            logger.info("=" * 60)

            if not ULTRALYTICS_AVAILABLE:
                raise RuntimeError("Ultralytics not available")
            if not PEOPLE_COUNTING_AVAILABLE:
                raise RuntimeError("People counting system not available")

            device = setup_device()
            model_path = find_model_file(model_dir)
            weights_path = find_osnet_weights(model_dir)

            file_size = os.path.getsize(model_path) / 1024 ** 2
            logger.info(f"Loading YOLO model: {os.path.basename(model_path)} ({file_size:.1f}MB)")

            yolo_model = YOLO(model_path)
            yolo_model.to(device)

            logger.info("Initializing People Counting System Manager")
            config = {
                "confidence_threshold": 0.35,
                "device": str(device),
                "max_stored_features": 30,
                "similarity_threshold": 0.55,
                "osnet_weights_path": weights_path
            }

            system_manager = PeopleCountingSystemManager(yolo_model, config)
            model_loaded = True

            gpu_info = {}
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
                }

            response = {
                "status": "loaded",
                "device": str(device),
                "model_size_mb": round(file_size, 1),
                "gpu_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
                "model_name": "people8s.pt",
                "reid_enabled": True,
                "reid_threshold": config["similarity_threshold"],
                "osnet_weights": weights_path if weights_path else "Not found (using random init)",
                "alert_debouncing": "ENABLED (State-based)",
                **gpu_info
            }

            logger.info(f"Model loading complete: {response}")
            return response

        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            logger.error(error_msg)

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
    """
    Parse and validate input.

    SIMPLIFIED:
    - alert_rate: Critical alert (user controlled)
    - Warning: FIXED at 65% (not user controlled)
    - confidence_threshold: OPTIONAL (defaults to 0.35)
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    try:
        data = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Required fields
    required = ["camid", "org_id", "userid", "encoding", "threshold", "alert_rate",
                "return_annotated"]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Type conversion and validation
    try:
        data["camid"] = int(data["camid"])
        data["org_id"] = int(data["org_id"])
        data["userid"] = int(data["userid"])
        data["threshold"] = int(data["threshold"])
        data["alert_rate"] = int(data["alert_rate"])
        data["return_annotated"] = bool(data["return_annotated"])
        data["confidence_threshold"] = float(data.get("confidence_threshold", 0.35))

        # Validate ranges
        if not (0.01 <= data["confidence_threshold"] <= 0.99):
            raise ValueError("confidence_threshold must be between 0.01 and 0.99")
        if not (0 <= data["alert_rate"] <= 100):
            raise ValueError("alert_rate must be between 0 and 100")
        if data["threshold"] < 1:
            raise ValueError("threshold must be >= 1")
        if not data["encoding"]:
            raise ValueError("encoding cannot be empty")

    except (ValueError, TypeError) as e:
        raise ValueError(f"Validation failed: {e}")

    return data


def predict_fn(input_data: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Main prediction function with state-based alert debouncing"""
    start_time = time.time()

    if not model_loaded or system_manager is None:
        return create_error_response(input_data, "Model not initialized", start_time)

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

        # Process frame (warning at fixed 65%)
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

        # Add performance metrics
        processing_time = (time.time() - start_time) * 1000
        gpu_stats = get_gpu_stats()

        # Log metrics
        metrics = {
            "processing_time_ms": processing_time,
            "people_detected": result.get("Total_people_detected", 0),
            "gpu_memory_percent": gpu_stats["gpu_memory_percent"],
            "gpu_utilization_percent": gpu_stats["gpu_utilization_percent"],
            "alert_triggered": result.get("is_alert_triggered", False)
        }
        log_metrics_to_console(metrics, camid, org_id)

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
        "Exit_time": [],
        "exitid": [],
        "People_dwell_time": [],
        "Confidence_scores": [],
        "Bounding_boxes": [],
        "x": [], "y": [], "w": [], "h": [],
        "accuracy": [],
        "Total_entries": 0,
        "Total_exits": 0,
        "Net_count": 0,
        "Occupancy_percentage": 0.0,
        "Over_capacity_count": 0,
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
    logger.info("=" * 60)
    logger.info("People Counting Inference Handler - Updated Version")
    logger.info("=" * 60)
    logger.info("Features:")
    logger.info("  - State-based alert debouncing (inspired by queue monitoring)")
    logger.info("  - Integer alert_rate")
    logger.info("  - Optional confidence_threshold (default: 0.35)")
    logger.info("  - Corrected Status logic: 'Alert', 'Warning', ''")
    logger.info(f"Dependencies - PyTorch: {TORCH_AVAILABLE}, Ultralytics: {ULTRALYTICS_AVAILABLE}")
    if TORCH_AVAILABLE:
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info("=" * 60)