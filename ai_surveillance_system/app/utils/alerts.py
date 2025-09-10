import os
import csv
from typing import Dict, List

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALERTS_CSV = os.path.join(BASE, "data", "alerts.csv")
os.makedirs(os.path.dirname(ALERTS_CSV), exist_ok=True)
if not os.path.exists(ALERTS_CSV):
    with open(ALERTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "matched_name", "video", "frame_no", "similarity", "saved_image"])

def append_alert(alert: Dict):
    with open(ALERTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([alert.get("timestamp"), alert.get("matched_name"),
                         alert.get("video"), alert.get("frame_no"),
                         alert.get("similarity"), alert.get("saved_image")])

def read_alerts() -> List[Dict]:
    rows = []
    with open(ALERTS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows
