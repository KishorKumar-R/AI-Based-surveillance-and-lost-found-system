import sys
import os

print("[DEBUG] Current working directory:", os.getcwd())  # Optional debug

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.face_recognition.embedder import load_model, preprocess_image, get_face_embedding
from app.face_recognition.matcher import find_best_match

if __name__ == "__main__":
    image_path = r"D:\Projects\Ujjain\ai_surveillance_system\data\missing_faces\Elonmiss.jpg"
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "face_db.pkl"))

    model = load_model()
    image = preprocess_image(image_path)
    embedding, found = get_face_embedding(model, image)

    if found:
        match = find_best_match(embedding, db_path, threshold=0.5)
        print("Final Match Result:", match)
    else:
        print("No face found in the query image.")
