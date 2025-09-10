import pickle
import numpy as np
from typing import Optional

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def find_best_match(query_embedding: np.ndarray, database_path: str, threshold: float = 0.5) -> Optional[str]:
    with open(database_path, "rb") as f:
        face_db = pickle.load(f)

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

"""
# Uncomment the below block to test matcher.py independently

if __name__ == "__main__":
    import os
    from embedder import load_model, preprocess_image, get_face_embedding

    sample_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "missing_faces", "Elonmiss.jpg"))
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "face_db.pkl"))

    print(f"[DEBUG] Image Path: {sample_img_path}")
    print(f"[DEBUG] DB Path: {db_path}")

    model = load_model()
    image = preprocess_image(sample_img_path)
    embedding, found = get_face_embedding(model, image)

    if found:
        result = find_best_match(embedding, db_path, threshold=0.5)
        print(f"[RESULT] Match: {result}")
    else:
        print("[ERROR] No face detected in the image.")
"""

