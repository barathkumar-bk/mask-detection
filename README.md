import cv2
import time
import os
import pygame
from ultralytics import YOLO
from datetime import datetime
import threading

# === CONFIGURATION ===
CUSTOM_MODEL_PATH = r"mask model_path" # Your trained model
ALARM_SOUND_PATH = r"Audio_path"  # Your alert sound
CONFIDENCE_THRESHOLD = 0.5

# === INITIALIZE ===
pygame.mixer.init()

# Load models
if os.path.exists(CUSTOM_MODEL_PATH):
    mask_model = YOLO(CUSTOM_MODEL_PATH)
    print("✅ Custom mask model loaded.")
else:
    print("❌ Mask model not found.")
    exit()

person_model = YOLO("yolov8n.pt")
print("✅ Fallback YOLOv8n loaded.")

# Load sound
def play_alarm():
    def play():
        try:
            pygame.mixer.music.load(ALARM_SOUND_PATH)
            pygame.mixer.music.play()
            time.sleep(2)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Sound error: {e}")
    threading.Thread(target=play, daemon=True).start()

# === DETECTION LOOP ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not available")
    exit()

print("🎥 Camera started. Press 'q' to quit.")

last_alert_time = 0
alert_delay = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to grab frame")
        break

    # Step 1: Detect using custom mask model
    results_mask = mask_model(frame, conf=CONFIDENCE_THRESHOLD)
    has_mask = False

    for result in results_mask:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Assuming class 0 = mask
                has_mask = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "MASK", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Step 2: If no mask found, detect unmasked people with YOLOv8n
    if not has_mask:
        results_person = person_model(frame, conf=CONFIDENCE_THRESHOLD)
        person_detected = False
        for result in results_person:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # person
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "NO MASK", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if person_detected:
            # Warning banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "⚠ ALERT: NO MASK DETECTED!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # Sound alert
            if time.time() - last_alert_time > alert_delay:
                play_alarm()
                last_alert_time = time.time()

    cv2.imshow("Mask Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("🧹 Done.")
