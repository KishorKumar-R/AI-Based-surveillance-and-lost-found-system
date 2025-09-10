import os
import cv2
import pickle
import numpy as np
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from insightface.app import FaceAnalysis
from app.face_recognition.matcher import find_best_match

# Paths
videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "videos"))
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "face_db.pkl"))

# Load database
with open(db_path, "rb") as f:
    face_db = pickle.load(f)
print(f"[INFO] Loaded {len(face_db)} enrolled faces from database.")

# Prepare model
model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0, det_size=(640, 640))

# Config
FRAME_SKIP = 5   # analyze every 5th frame
TARGET_FPS = 60  # simulate 60fps playback
FRAME_TIME = 1.0 / TARGET_FPS

last_faces = []

# Loop over all videos
for file_name in os.listdir(videos_dir):
    if not file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    video_path = os.path.join(videos_dir, file_name)
    print(f"[INFO] Processing video: {file_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open {file_name}. Skipping.")
        continue

    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 360))

        if frame_count % FRAME_SKIP == 0:
            faces = model.get(frame)
            last_faces = []
            for face in faces:
                embedding = face.embedding
                match_name = find_best_match(embedding, db_path, threshold=0.5)
                box = face.bbox.astype(int)
                last_faces.append((box, match_name))

        for box, name in last_faces:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition (Smooth 60fps)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Enforce ~60fps playback
        elapsed = time.time() - start_time
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

    cap.release()
    print(f"[INFO] Finished processing {file_name}")

cv2.destroyAllWindows()
