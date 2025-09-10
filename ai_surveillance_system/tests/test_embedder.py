import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.face_recognition.embedder import (
    load_model,
    preprocess_image,
    get_face_embedding
)

def test_face_embedding(image_path: str):
    model = load_model()
    image = preprocess_image(image_path)
    embedding, found = get_face_embedding(model, image)

    if found:
        print("Face detected successfully.")
        print("Embedding shape:", embedding.shape)
        print("First 10 embedding values:", embedding[:10])
    else:
        print("No face detected in the image.")

if __name__ == "__main__":
    test_face_embedding("data/missing_faces/testimg1.jpg")
