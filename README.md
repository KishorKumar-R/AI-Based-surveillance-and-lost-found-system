# AI-Based Surveillance and Lost & Found System

## Overview
This project is an AI-powered surveillance and crowd management system designed for large-scale events. 
It integrates real-time face recognition, missing person detection, and predictive crowd flow analysis 
to enhance public safety and assist authorities in managing crowds efficiently.  

---

## Features
- **Real-time Face Recognition:** Identify known individuals from CCTV or video feeds.  
- **Missing Person Detection:** Detect and track missing persons in crowds.  
- **Lost & Found System:** Match and locate lost items and individuals using AI-based matching.  
- **Crowd Flow Analysis:** Predict and visualize crowd movements to optimize safety.  
- **Scalable Architecture:** Built using modular Python code, ONNX, PyTorch, and Flask.  

---

## Tech Stack
- **Languages:** Python 3.12  
- **Frameworks:** Flask, FastAPI (for APIs), PyTorch  
- **Libraries:** OpenCV, Face Recognition, NumPy, Pandas  
- **Models:** ArcFace, YOLOv8-face, custom prediction models  
- **Others:** Docker for containerization, GitHub for version control  

---

## Folder Structure
ai_surveillance_system/
├── app/ # Core application modules
│ ├── api/ # API routes
│ ├── detection/ # Object & face detection
│ ├── face_recognition/ # Face embeddings, matcher, enrollment
│ ├── prediction/ # Crowd prediction models
│ └── utils/ # Helper functions & logger
├── data/ # Input videos, frames, embeddings, missing/enrolled faces
├── docker/ # Docker setup files
├── models/ # Pretrained models
├── notebooks/ # Demo & training notebooks
├── scripts/ # Helper scripts
├── tests/ # Unit tests
├── main.py # Entry point
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy code

---

## Installation
1. Clone the repository:
bash
git clone https://github.com/<your-username>/AI-Based-surveillance-and-lost-found-system.git
cd AI-Based-surveillance-and-lost-found-system
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the system:

bash
python main.py

Usage:

Face Recognition: Add enrolled face images to data/enrolled_faces/ and run the recognition module.

Missing Person Detection: Place missing person images in data/missing_faces/ for search.

Crowd Flow Prediction: Use sample input videos in data/videos/ to test predictive analysis.

Dashboard: Access visualizations through the Flask dashboard (app/dashboard/app.py).

Future Enhancements:

Multi-camera integration for large-scale events

Mobile or web app interface for real-time monitoring

Automated alert system using SMS/email when missing persons are detected

Improved predictive algorithms for crowd flow using AI

License
This project is licensed under the MIT License.

Contact
Developed by Kishor Kumar
GitHub: https://github.com/KishorKumar-R
