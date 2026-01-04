import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def extract_measurements(image_path, height_cm):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("❌ Image not found or path incorrect.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect pose
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=False, model_complexity=2) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            raise Exception("❌ No pose detected. Ensure full-body image with visible person.")

        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        # Helper to convert normalized coordinates to pixels
        def get_point(lm):
            return int(lm.x * w), int(lm.y * h)

        # Important keypoints
        left_shoulder = get_point(landmarks[11])
        right_shoulder = get_point(landmarks[12])
        left_hip = get_point(landmarks[23])
        right_hip = get_point(landmarks[24])
        left_ankle = get_point(landmarks[27])
        right_ankle = get_point(landmarks[28])
        left_chest = get_point(landmarks[11])
        right_chest = get_point(landmarks[12])

        # Get overall body height in pixels (head to ankle midpoint)
        top_head = get_point(landmarks[0])
        mid_ankle = ((left_ankle[0] + right_ankle[0]) // 2, (left_ankle[1] + right_ankle[1]) // 2)
        person_pixel_height = get_distance(top_head, mid_ankle)

        # Convert pixel to cm
        cm_per_pixel = height_cm / person_pixel_height

        # Shoulder width
        shoulder_px = get_distance(left_shoulder, right_shoulder)
        shoulder_cm = (shoulder_px * cm_per_pixel)-10

        # Chest width (approx between mid shoulder line)
        chest_px = get_distance(left_chest, right_chest)
        chest_cm = (chest_px * cm_per_pixel)+10

        # Waist width (approx between hip points)
        waist_px = get_distance(left_hip, right_hip)
        waist_cm = waist_px * cm_per_pixel*2

        # Front Length (shoulder to hip midpoint)
        mid_hip = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
        front_length_px = get_distance(((left_shoulder[0]+right_shoulder[0])//2,
                                        (left_shoulder[1]+right_shoulder[1])//2),
                                       mid_hip)
        front_length_cm = (front_length_px * cm_per_pixel)+10

        # Draw pose landmarks
        annotated = image.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite("pose_detected.jpg", annotated)

        # Return all measurements
        return {
            "Shoulder(cm)": round(shoulder_cm, 2),
            "Chest(cm)": round(chest_cm, 2),
            "Waist(cm)": round(waist_cm, 2),
            "Front Length(cm)": round(front_length_cm, 2)
        }

if __name__ == "__main__":
    image_path = input("Enter image filename (e.g., test_image.jpg): ")
    height_cm = float(input("Enter your height in cm: "))
    # weight = input("Enter your weight (optional, not used): ")
    # age = input("Enter your age (optional, not used): ")

    try:
        measures = extract_measurements(image_path, height_cm)
        print("\n✅ Extracted Body Measurements (in cm):")
        for k, v in measures.items():
            print(f"{k}: {v}")
        print("\n📸 Annotated image saved as 'pose_detected.jpg'")
    except Exception as e:
        print(str(e))
