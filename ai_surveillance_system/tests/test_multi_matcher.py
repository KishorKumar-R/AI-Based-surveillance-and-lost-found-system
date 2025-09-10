import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def find_best_match(query_embedding: np.ndarray, face_db: dict, threshold: float = 0.5) -> str:
    best_match = "Unknown"
    best_score = -1

    for name, db_embedding in face_db.items():
        similarity = cosine_similarity(query_embedding, db_embedding)
        if similarity > best_score:
            best_score = similarity
            best_match = name

    if best_score >= threshold:
        print(f"[MATCH] {best_match} (Similarity: {best_score:.3f})")
        return best_match
    else:
        print(f"[NO MATCH] Best similarity: {best_score:.3f}")
        return "Unknown"

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "face_db.pkl"))
missing_faces_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "missing_faces"))

with open(db_path, "rb") as f:
    face_db = pickle.load(f)

print(f"[INFO] Loaded {len(face_db)} enrolled faces from database.")

model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0, det_size=(640, 640))

for image_name in os.listdir(missing_faces_dir):
    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print(f"\n[INFO] Processing image: {image_name}")
    image_path = os.path.join(missing_faces_dir, image_name)
    image = cv2.imread(image_path)

    faces = model.get(image)
    print(f"[INFO] Detected {len(faces)} faces.")

    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            print(f"[WARNING] Face crop {idx + 1} is empty. Skipping.")
            continue

        emb = face.embedding
        match_name = find_best_match(emb, face_db, threshold=0.5)
        print(f"Face {idx + 1}: {match_name}")
