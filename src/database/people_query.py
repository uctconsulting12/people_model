# import json
# import logging
# import psycopg2
# import os
# logger = logging.getLogger("detection")
# logger.setLevel(logging.INFO)

# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# # PostgreSQL connection
# try:
#     conn = psycopg2.connect(
#         host=os.environ["DB_HOST"],
#         dbname=os.environ["DB_NAME"],
#         user=os.environ["DB_USER"],
#         password=os.environ["DB_PASSWORD"],
#         port=int(os.environ.get("DB_PORT", 5432))
#     )
#     conn.autocommit = True
#     cursor = conn.cursor()
#     logger.info("✅ Connected to PostgreSQL")
# except Exception as e:
#     logger.error(f"Failed to connect to PostgreSQL: {e}")
#     raise




# def insert_people_counting(data, s3_url):
#     """Insert detection data into detections table"""
#     try:
#         with conn.cursor() as cursor:
#             insert_query = """
#                 INSERT INTO people_counting (
#                     camid, frame_id, time_stamp, frame_count,
#                     total_people_detected, current_occupancy,
#                     people_ids, entry_time, people_dwell_time,
#                     confidence_scores, bounding_boxes, x, y, w, h,
#                     accuracy, total_entries, total_exits, net_count,
#                     occupancy_percentage, average_dwell_time, max_occupancy,
#                     status, is_alert_triggered, processing_status,
#                     org_id, userid, s3_url
#                 ) VALUES (
#                     %s, %s, %s, %s,
#                     %s, %s,
#                     %s, %s, %s,
#                     %s, %s, %s, %s, %s, %s,
#                     %s, %s, %s, %s,
#                     %s, %s, %s,
#                     %s, %s, %s,
#                     %s, %s, %s
#                 )
#                 ON CONFLICT (frame_id) DO UPDATE SET
#                     time_stamp = EXCLUDED.time_stamp,
#                     total_people_detected = EXCLUDED.total_people_detected,
#                     current_occupancy = EXCLUDED.current_occupancy,
#                     people_ids = EXCLUDED.people_ids,
#                     entry_time = EXCLUDED.entry_time,
#                     people_dwell_time = EXCLUDED.people_dwell_time,
#                     confidence_scores = EXCLUDED.confidence_scores,
#                     bounding_boxes = EXCLUDED.bounding_boxes,
#                     x = EXCLUDED.x,
#                     y = EXCLUDED.y,
#                     w = EXCLUDED.w,
#                     h = EXCLUDED.h,
#                     accuracy = EXCLUDED.accuracy,
#                     occupancy_percentage = EXCLUDED.occupancy_percentage,
#                     average_dwell_time = EXCLUDED.average_dwell_time,
#                     status = EXCLUDED.status,
#                     is_alert_triggered = EXCLUDED.is_alert_triggered,
#                     processing_status = EXCLUDED.processing_status,
#                     s3_url = EXCLUDED.s3_url;
#             """

#             cursor.execute(insert_query, (
#                 data['camid'],
#                 data['Frame_Id'],
#                 data['Time_stamp'],
#                 data['Frame_Count'],
#                 data['Total_people_detected'],
#                 data['Current_occupancy'],
#                 json.dumps(data['People_ids']),
#                 json.dumps(data['Entry_time']),
#                 json.dumps(data['People_dwell_time']),
#                 json.dumps(data['Confidence_scores']),
#                 json.dumps(data['Bounding_boxes']),
#                 json.dumps(data['x']),
#                 json.dumps(data['y']),
#                 json.dumps(data['w']),
#                 json.dumps(data['h']),
#                 json.dumps(data['accuracy']),
#                 data['Total_entries'],
#                 data['Total_exits'],
#                 data['Net_count'],
#                 data['Occupancy_percentage'],
#                 data['Average_dwell_time'],
#                 data['Max_occupancy'],
#                 data['Status'],
#                 data['is_alert_triggered'],
#                 data['Processing_Status'],
#                 data['org_id'],
#                 data['userid'],
#                 s3_url
#             ))

#             conn.commit()
#             logger.info(f"✅ Detection inserted for frame: {data['Frame_Id']}")
#             return True

#     except Exception as e:
#         conn.rollback()
#         logger.error(f"❌ Failed to insert detection: {e}")
#         return False


import json
import logging
import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)

load_dotenv()

# ---------------------------------------------------------
# 1️⃣  FUNCTION: SETUP DATABASE + CONNECTION POOL
# ---------------------------------------------------------
def setup_database():
    """
    Create PostgreSQL connection pool
    """
    try:
        pool = SimpleConnectionPool(
            1, 20,    # min 1, max 20 connections
            host=os.environ["DB_HOST"],
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            port=int(os.environ.get("DB_PORT", 5432))
        )
        logger.info("✅ Connection Pool created successfully")
        return pool
    except Exception as e:
        logger.error(f"❌ Failed to create connection pool: {e}")
        raise


# Global pool object
pool = setup_database()

# ---------------------------------------------------------
# 2️⃣ FUNCTION: INSERT PEOPLE COUNTING
# ---------------------------------------------------------
def insert_people_counting(data, s3_url):
    """
    Insert/update a record in people_counting table
    using PostgreSQL connection pool
    """
    conn = pool.getconn()

    insert_query = """
        INSERT INTO people_counting (
            camid, frame_id, time_stamp, frame_count,
            total_people_detected, current_occupancy,
            people_ids, entry_time, people_dwell_time,
            confidence_scores, bounding_boxes, x, y, w, h,
            accuracy, total_entries, total_exits, net_count,
            occupancy_percentage, average_dwell_time, max_occupancy,
            status, is_alert_triggered, processing_status,
            org_id, userid, s3_url
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (frame_id) DO UPDATE SET
            time_stamp = EXCLUDED.time_stamp,
            total_people_detected = EXCLUDED.total_people_detected,
            current_occupancy = EXCLUDED.current_occupancy,
            people_ids = EXCLUDED.people_ids,
            entry_time = EXCLUDED.entry_time,
            people_dwell_time = EXCLUDED.people_dwell_time,
            confidence_scores = EXCLUDED.confidence_scores,
            bounding_boxes = EXCLUDED.bounding_boxes,
            x = EXCLUDED.x,
            y = EXCLUDED.y,
            w = EXCLUDED.w,
            h = EXCLUDED.h,
            accuracy = EXCLUDED.accuracy,
            occupancy_percentage = EXCLUDED.occupancy_percentage,
            average_dwell_time = EXCLUDED.average_dwell_time,
            status = EXCLUDED.status,
            is_alert_triggered = EXCLUDED.is_alert_triggered,
            processing_status = EXCLUDED.processing_status,
            s3_url = EXCLUDED.s3_url;
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(insert_query, (
                data['camid'],
                data['Frame_Id'],
                data['Time_stamp'],
                data['Frame_Count'],
                data['Total_people_detected'],
                data['Current_occupancy'],
                json.dumps(data['People_ids']),
                json.dumps(data['Entry_time']),
                json.dumps(data['People_dwell_time']),
                json.dumps(data['Confidence_scores']),
                json.dumps(data['Bounding_boxes']),
                json.dumps(data['x']),
                json.dumps(data['y']),
                json.dumps(data['w']),
                json.dumps(data['h']),
                json.dumps(data['accuracy']),
                data['Total_entries'],
                data['Total_exits'],
                data['Net_count'],
                data['Occupancy_percentage'],
                data['Average_dwell_time'],
                data['Max_occupancy'],
                data['Status'],
                data['is_alert_triggered'],
                data['Processing_Status'],
                data['org_id'],
                data['userid'],
                s3_url
            ))

        conn.commit()
        logger.info(f"✅ Inserted/Updated: Frame {data['Frame_Id']}")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Database insert error: {e}")
        return False

    finally:
        pool.putconn(conn)

