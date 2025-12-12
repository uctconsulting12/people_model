from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.websocket.people_w_local import run_peoplecounting_detection


from src.handlers.people_handler import people_websocket_handler




from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores --------------
queue_sessions = {}
people_sessions = {}


detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- People Counting WebSocket ----------------
@app.websocket("/ws/people_counting/{client_id}")
async def websocket_people(ws: WebSocket, client_id: str):
    await people_websocket_handler(detection_executor,storage_executor,ws, client_id, people_sessions, run_peoplecounting_detection, "PeopleCounting")






