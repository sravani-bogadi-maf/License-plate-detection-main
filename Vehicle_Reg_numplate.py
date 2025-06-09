from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import re

# Load YOLO for license plate detection and PaddleOCR for recognition
yolo_model = YOLO("license_plate_detector.pt")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

def calculate_sharpness(image):
    """Calculate the sharpness of an image using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_valid_plate(text):
    """Checks if the detected plate follows a valid format."""
    return bool(re.match(r"^[A-Z0-9]{5,10}$", text))  # Strict regex filter

def is_same_car(existing_bbox, new_bbox, iou_threshold=0.4, shift_tolerance=50):
    """Check if two bounding boxes belong to the same car using IoU and positional shift."""
    x1_e, y1_e, x2_e, y2_e = existing_bbox
    x1_n, y1_n, x2_n, y2_n = new_bbox

    if abs(x1_n - x1_e) < shift_tolerance and abs(y1_n - y1_e) < shift_tolerance:
        return True  # Small shifts are allowed

    inter_x1 = max(x1_e, x1_n)
    inter_y1 = max(y1_e, y1_n)
    inter_x2 = min(x2_e, x2_n)
    inter_y2 = min(y2_e, y2_n)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    existing_area = (x2_e - x1_e) * (y2_e - y1_e)
    new_area = (x2_n - x1_n) * (y2_n - y1_n)

    iou = inter_area / float(existing_area + new_area - inter_area)
    return iou > iou_threshold

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 2)
    frame_count = 0
    unique_texts = set()
    car_tracker = defaultdict(lambda: {"last_seen": None, "best_sharpness": 0, "best_frame": None, "plate_img": None, "conf": 0})
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = yolo_model(frame)
            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    if conf < 0.7:
                        continue

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    license_plate = frame[y1:y2, x1:x2]
                    if license_plate.shape[0] < 15 or license_plate.shape[1] < 40:
                        continue

                    sharpness = calculate_sharpness(license_plate)
                    found_match = False

                    for existing_car in list(car_tracker.keys()):
                        existing_bbox = tuple(map(int, existing_car.split('-')))
                        if is_same_car(existing_bbox, (x1, y1, x2, y2)):
                            found_match = True
                            if sharpness > car_tracker[existing_car]["best_sharpness"] or conf > car_tracker[existing_car]["conf"]:
                                car_tracker[existing_car] = {"last_seen": datetime.now(), "best_sharpness": sharpness, "best_frame": frame.copy(), "plate_img": license_plate, "conf": conf}
                            break

                    if not found_match:
                        car_id = f"{x1}-{y1}-{x2}-{y2}"
                        car_tracker[car_id] = {"last_seen": datetime.now(), "best_sharpness": sharpness, "best_frame": frame.copy(), "plate_img": license_plate, "conf": conf}
        
        frame_count += 1
    
    cap.release()
    return car_tracker

def extract_plate_numbers(car_tracker):
    final_detected_plates = {}
    if car_tracker:
        for car_id, data in car_tracker.items():
            if data["best_frame"] is None:
                continue
            plate = data["plate_img"]
            frame = data["best_frame"]
            ocr_results = ocr.predict(plate)
            if ocr_results:
                # The structure of ocr_results has changed in the new version
                detected_texts = []
                for line in ocr_results:
                    for word_info in line:
                        if word_info and len(word_info) >= 2:
                            # Extract text and confidence from the word_info
                            text = word_info[0]
                            confidence = word_info[1]
                            detected_texts.append((text, confidence))
                if detected_texts:
                    final_text, final_conf = max(detected_texts, key=lambda x: x[1])
                    if is_valid_plate(final_text):
                        final_detected_plates[car_id] = (final_text, final_conf, frame)
    return final_detected_plates

def display_results(final_detected_plates):
    if final_detected_plates:
        for car_id, (final_text, final_conf, frame) in final_detected_plates.items():
            x1, y1, x2, y2 = map(int, car_id.split('-'))
            print("✅ Final Detected License Plate:", final_text)
            cv2.putText(frame, final_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
    else:
        print("❌ No valid license plate detected.")

if __name__ == "__main__":
    video_path = "V4.mp4"
    car_tracker = process_video(video_path)
    final_detected_plates = extract_plate_numbers(car_tracker)
    display_results(final_detected_plates)