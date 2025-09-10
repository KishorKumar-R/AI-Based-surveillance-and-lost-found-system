import os
import cv2
import pickle
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from insightface.app import FaceAnalysis
from app.face_recognition.embedder import get_face_embedding
from app.face_recognition.matcher import find_best_match

video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data1", "videos", "Sample.mp4"))
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "face_db.pkl"))

with open(db_path, "rb") as f:
    face_db = pickle.load(f)
print(f"[INFO] Loaded {len(face_db)} enrolled faces from database.")

model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Failed to open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = model.get(frame)
    for face in faces:
        embedding = face["embedding"]
        match_name = find_best_match(embedding, db_path, threshold=0.5)

        box = face["bbox"].astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, match_name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

