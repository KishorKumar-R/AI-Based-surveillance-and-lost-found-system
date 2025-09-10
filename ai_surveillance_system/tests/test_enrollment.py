import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.face_recognition.enrollment import build_face_database

if __name__ == "__main__":
    enrolled_faces_dir = "data/enrolled_faces"
    output_path = "face_db.pkl"

    build_face_database(enrolled_faces_dir, output_path)
