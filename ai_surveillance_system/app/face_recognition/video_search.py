import os
import cv2
import time
import pickle
import numpy as np
from typing import List, Dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

def annotate_and_save(frame, bbox, name, out_path):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{name}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(out_path, frame)

def search_in_videos(query_image_path: str,
                     videos_dir: str,
                     outputs_dir: str,
                     model,
                     db_path: str,
                     threshold: float = 0.55,
                     skip_frames: int = 5) -> List[Dict]:
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)

    # prepare query embedding
    from app.face_recognition.embedder import preprocess_image
    q_img = preprocess_image(query_image_path)
    q_faces = model.get(q_img)
    if not q_faces:
        return []

    q_emb = q_faces[0].embedding

    results = []
    videos = [f for f in os.listdir(videos_dir) if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]
    for video in videos:
        path = os.path.join(videos_dir, video)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue

        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if frame_no % skip_frames != 0:
                continue

            small = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
            faces = model.get(small)
            for face in faces:
                emb = face.embedding
                sim = cosine_similarity(q_emb, emb)
                if sim >= threshold:
                    bbox = face.bbox.astype(int).tolist()
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_file = f"match_{os.path.splitext(video)[0]}_{frame_no}_{int(sim*1000)}.jpg"
                    out_path = os.path.join(outputs_dir, out_file)
                    annotate_and_save(small.copy(), bbox, f"match ({sim:.3f})", out_path)
                    results.append({
                        "matched_name": "Unknown",
                        "video": video,
                        "frame_no": frame_no,
                        "similarity": float(sim),
                        "saved_image": out_path,
                        "timestamp": ts
                    })
            # end faces
        cap.release()
    return results
