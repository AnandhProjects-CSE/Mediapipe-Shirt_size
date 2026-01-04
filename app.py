from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
import uuid
from functools import wraps
import pandas as pd
import gc
# ---------- Config ----------
UPLOAD_FOLDER = "static/output"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- App ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "Rajkumar@6622"  # Change this!

# Simple user database (in production, use proper database)
USERS = {
    "admin": "admin123",
    "demo": "demo123",
    "pavan": "pavan123"
}

# ---------- Load models ----------

try:
    rf = joblib.load("random_forest_size_model.pkl")
    le = joblib.load("label_encoder.pkl")
    xgb = joblib.load("xgboost_size_model.pkl")
    if not hasattr(xgb, "use_label_encoder"):
        xgb.use_label_encoder = False
except Exception:
    xgb = None


# ---------- Mediapipe setup ----------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True,model_complexity=0,enable_segmentation=False,min_detection_confidence=0.5)

# ---------- Authentication Decorator ----------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def to_px(lm, w, h):
    return int(max(0, min(w-1, lm.x * w))), int(max(0, min(h-1, lm.y * h)))

def pixel_dist(a, b):
    return float(((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5)

def extract_measurements_from_image(img_bgr, user_height_cm=170.0, show_img=False):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None, None

    lm = res.pose_landmarks.landmark

    # normalized x-distance measurements
    left_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    # normalized width in fraction of image width (0..1)
    shoulder_norm = abs(right_sh.x - left_sh.x)
    chest_norm = abs(right_el.x - left_el.x)
    waist_norm = abs(right_hip.x - left_hip.x)

    # Convert normalized to cm using provided height
    shoulder_cm = shoulder_norm * user_height_cm
    chest_cm = chest_norm * user_height_cm
    waist_cm = waist_norm * user_height_cm

    # Apply final conversion rules
    shoulder_final = shoulder_cm
    chest_final = (chest_cm * 4.0) - 10.0
    waist_final = (waist_cm * 4.0) - 10.0

    # Annotate image for display
    annotated = img_bgr.copy()
    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # draw horizontal lines for visual cue
    try:
        left_sh_px = to_px(left_sh, w, h)
        right_sh_px = to_px(right_sh, w, h)
        left_el_px = to_px(left_el, w, h)
        right_el_px = to_px(right_el, w, h)
        left_hip_px = to_px(left_hip, w, h)
        right_hip_px = to_px(right_hip, w, h)

        cv2.line(annotated, left_sh_px, right_sh_px, (0,255,0), 3)
        cv2.line(annotated, left_el_px, right_el_px, (0,165,255), 3)
        cv2.line(annotated, left_hip_px, right_hip_px, (255,0,0), 3)
    except Exception:
        pass

    return (shoulder_final, chest_final, waist_final), annotated

# ---------- Routes ----------
@app.route("/")
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if username in USERS and USERS[username] == password:
            session['username'] = username
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('username', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route("/shirt-size-1", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        try:
            if "image" not in request.files:
                flash("No file uploaded", "error")
                return render_template("index.html", error="No file part")
            
            file = request.files["image"]
            if file.filename == "":
                flash("No file selected", "error")
                return render_template("index.html", error="No selected file")
            
            if file and allowed_file(file.filename):
                # save file
                ext = file.filename.rsplit(".", 1)[1].lower()
                fname = f"{uuid.uuid4().hex}.{ext}"
                fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                file.save(fpath)

                # get user height
                try:
                    height_cm = float(request.form.get("height_cm", "170"))
                except:
                    height_cm = 170.0

                # process image
                img = cv2.imread(fpath)
                img = cv2.resize(img, (640, 480))
                measurements, annotated = extract_measurements_from_image(img, user_height_cm=height_cm)
                del img 
                gc.collect()
                if measurements is None:
                    flash("No person detected in image", "error")
                    return render_template("index.html", error="No person/pose detected. Upload a full-body frontal image.")

                shoulder_final, chest_final, waist_final = measurements

                # Prepare dataframe for model
                
                test_df = pd.DataFrame([[shoulder_final, chest_final, waist_final]],
                                    columns=["shoulder", "chest_cm", "waist_cm"])

                # Predict
                rf_pred = rf.predict(test_df)[0]
                rf_size = le.inverse_transform([rf_pred])[0]
                
                if xgb is not None:
                    xgb_pred = xgb.predict(test_df)[0]
                    xgb_size = le.inverse_transform([xgb_pred])[0]
                else:
                    xgb_size = None
                del test_df
                gc.collect()
                # Final decision
                final_size = rf_size if (xgb_size is None or rf_size == xgb_size) else rf_size

                # save annotated image
                out_name = f"annot_{uuid.uuid4().hex}.{ext}"
                out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
                cv2.imwrite(out_path, annotated)

                # show results
                return render_template("result.html",
                                    img_path=os.path.join("/", out_path),
                                    shoulder=round(shoulder_final, 2),
                                    chest=round(chest_final, 2),
                                    waist=round(waist_final, 2),
                                    rf_size=rf_size,
                                    xgb_size=xgb_size,
                                    final_size=final_size)
        except Exception as e:
            app.logger.error("Prediction failed", exc_info=True)
            flash("Prediction failed. Please try a smaller image.", "error")
            return render_template("index.html") 
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)