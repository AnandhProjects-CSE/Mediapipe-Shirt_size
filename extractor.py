import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_measurements(image_path):
    """
    Extracts shoulder, chest, waist pixel distances and person height (in pixels)
    from a full or half body image.
    Returns a dictionary with pixel distances.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the image.")

        lm = results.pose_landmarks.landmark

        def get_xy(idx):
            """Convert normalized coordinates to pixel coordinates."""
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # Key points
        left_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
        left_ankle = get_xy(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        left_elbow = get_xy(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        right_elbow = get_xy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        nose = get_xy(mp_pose.PoseLandmark.NOSE.value)

        # Person height in pixels (nose to ankle)
        ankles_y = max(left_ankle[1], right_ankle[1])
        person_pixel_height = ankles_y - nose[1]

        # Shoulder width
        shoulder_px = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

        # Chest width (approx mid of shoulders and hips)
        chest_y = int((left_shoulder[1] + left_hip[1]) / 2)
        chest_px = abs(right_shoulder[0] - left_shoulder[0]) * 0.85  # proportional chest width

        # Waist width (between hips)
        waist_px = np.linalg.norm(np.array(left_hip) - np.array(right_hip))

        # Optional debug visualization
        # cv2.line(img, left_shoulder, right_shoulder, (0, 255, 0), 2)
        # cv2.imshow('Pose', img); cv2.waitKey(0)

        return {
            "person_pixel_height": person_pixel_height,
            "shoulder_px": shoulder_px,
            "chest_px": chest_px,
            "waist_px": waist_px
        }

# Example test run
if __name__ == "__main__":
    path = "s12.png"  # change this to your test image
    try:
        result = extract_measurements(path)
        print("Measurements (pixels):", result)
    except Exception as e:
        print("Error:", e)
