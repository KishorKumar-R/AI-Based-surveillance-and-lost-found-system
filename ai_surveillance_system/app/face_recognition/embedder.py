import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import Tuple

def load_model():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_face_embedding(model: FaceAnalysis, image: np.ndarray) -> Tuple[np.ndarray, bool]:
    faces = model.get(image)
    if len(faces) == 0:
        return None, False
    return faces[0].embedding, True
