# # ===============================================
# # Final FIXED Model Training — Shirt Size Prediction
# # ===============================================

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# import joblib
# import numpy as np

# # ======== Step 1: Load Dataset ========
# df = pd.read_csv("dataset.csv")

# # Clean column names
# df.columns = [col.strip() for col in df.columns]

# # ======== Step 2: Clean Numeric Data ========
# for col in ['Chest(cm)', 'Front Length(cm)', 'Across Shoulder(cm)']:
#     df[col] = df[col].replace(' ', np.nan)
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Drop rows with missing data
# df.dropna(subset=['Chest(cm)', 'Front Length(cm)', 'Across Shoulder(cm)', 'Brand Size'], inplace=True)

# # ======== Step 3: Encode Categorical Labels ========
# le_brand_size = LabelEncoder()
# df['Brand Size'] = le_brand_size.fit_transform(df['Brand Size'])

# if 'Brand Name' in df.columns:
#     df['Brand Name'] = LabelEncoder().fit_transform(df['Brand Name'])
# if 'Type' in df.columns:
#     df['Type'] = LabelEncoder().fit_transform(df['Type'])

# # ======== Step 4: Features & Target ========
# X = df[['Chest(cm)', 'Front Length(cm)', 'Across Shoulder(cm)']]
# y = df['Brand Size']

# # ======== Step 5: Handle Few Samples ========
# if df['Brand Size'].value_counts().min() < 2:
#     print("⚠️ Some sizes have very few samples — stratify disabled for safe split.")
#     stratify_option = None
# else:
#     stratify_option = y

# # ======== Step 6: Train-Test Split ========
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=stratify_option
# )

# # Drop unseen labels in test
# mask = y_test.isin(np.unique(y_train))
# X_test = X_test[mask]
# y_test = y_test[mask]

# # ======== Step 7: Scale Data ========
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # ======== Step 8: Train Models ========
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# rf_acc = accuracy_score(y_test, rf.predict(X_test))

# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# xgb.fit(X_train, y_train)
# xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

# svm = SVC(kernel='rbf', gamma='auto', probability=True)
# svm.fit(X_train, y_train)
# svm_acc = accuracy_score(y_test, svm.predict(X_test))

# # ======== Step 9: Print Results ========
# print("\n✅ Model Accuracies:")
# print(f"Random Forest: {rf_acc*100:.2f}%")
# print(f"XGBoost: {xgb_acc*100:.2f}%")
# print(f"SVM: {svm_acc*100:.2f}%")

# # ======== Step 10: Save Models ========
# joblib.dump(rf, "random_forest_size_model_1.pkl")
# joblib.dump(xgb, "xgboost_size_model_1.pkl")
# joblib.dump(svm, "svm_size_model_1.pkl")
# joblib.dump(le_brand_size, "label_encoder_1.pkl")
# joblib.dump(scaler, "scaler.pkl")

# print("\n💾 Models and encoders saved successfully!")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===== Load dataset =====
data = pd.read_csv("dataset.csv")

# ===== Rename columns for consistency =====
data.rename(columns={
    "Across Shoulder(cm)": "shoulder",
    "Chest(cm)": "chest",
    "Front Length(cm)": "length",
    "Brand Size": "size"
}, inplace=True)

# ===== Clean and preprocess =====
data = data.dropna(subset=["shoulder", "chest", "length", "size"])
data["shoulder"] = pd.to_numeric(data["shoulder"], errors="coerce")
data["chest"] = pd.to_numeric(data["chest"], errors="coerce")
data["length"] = pd.to_numeric(data["length"], errors="coerce")
data = data.dropna()

X = data[["shoulder", "chest", "length"]]
y = data["size"]

# ===== Encode labels =====
le = LabelEncoder()
y = le.fit_transform(y)

# ===== Train-test split =====
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print("⚠️ Some sizes have very few samples — stratify disabled for safe split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# ===== Scaling =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== Random Forest =====
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ===== XGBoost (with fixed class_labels) =====
unique_classes = np.unique(y)
# xgb = XGBClassifier(
#     n_estimators=100,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric="mlogloss",
# )
# xgb.fit(X_train, y_train, classes=unique_classes)

# ===== Save models =====
# joblib.dump(rf, "random_forest_size_model.pkl")
# joblib.dump(xgb, "xgboost_size_model.pkl")
# joblib.dump(le, "label_encoder.pkl")
# joblib.dump(scaler, "scaler.pkl")

joblib.dump(rf, "random_forest_size_model_1.pkl")
# joblib.dump(xgb, "xgboost_size_model_1.pkl")
joblib.dump(le, "label_encoder_1.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models trained and saved successfully!")
print("Classes:", list(le.classes_))
