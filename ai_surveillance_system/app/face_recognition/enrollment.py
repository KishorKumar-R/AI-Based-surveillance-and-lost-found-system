import os
import pickle
import numpy as np
from typing import Dict
from app.face_recognition.embedder import load_model, preprocess_image, get_face_embedding

def build_face_database(input_dir: str, output_path: str) -> None:
    model = load_model()
    face_db: Dict[str, np.ndarray] = {}

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name = os.path.splitext(filename)[0]
        image_path = os.path.join(input_dir, filename)
        image = preprocess_image(image_path)

        embedding, found = get_face_embedding(model, image)
        if found:
            face_db[name] = embedding
            print(f"[INFO] Embedded: {name}")
        else:
            print(f"[WARNING] No face found in: {filename}")

    with open(output_path, "wb") as f:
        pickle.dump(face_db, f)
        print(f"[SUCCESS] Saved {len(face_db)} faces to: {output_path}")
