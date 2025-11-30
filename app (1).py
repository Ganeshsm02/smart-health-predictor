import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
from datetime import datetime

# ---------- App / DB setup ----------
app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "replace_this_with_secure_key")
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------- DB Models ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), nullable=False)
    disease = db.Column(db.String(60), nullable=False)
    probability = db.Column(db.Float, nullable=False)   # stored 0..1
    risk_level = db.Column(db.String(20), nullable=False)
    precautions = db.Column(db.Text, nullable=True)
    inputs = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ---------- Load models & scalers (joblib) ----------
MODEL_DIR = "models"
diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.joblib"))
diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_scaler.joblib"))

kidney_model = joblib.load(os.path.join(MODEL_DIR, "kidney_model.joblib"))
kidney_scaler = joblib.load(os.path.join(MODEL_DIR, "kidney_scaler.joblib"))

heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_model.joblib"))
heart_scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.joblib"))

liver_model = joblib.load(os.path.join(MODEL_DIR, "liver_model.joblib"))
liver_scaler = joblib.load(os.path.join(MODEL_DIR, "liver_scaler.joblib"))

# ---------- Feature lists (exact order expected by models) ----------
FEATURES = {
    "Diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "Kidney": [
        "age","bp","sg","al","su","bgr","bu","sc","sod","pot",
        "hemo","pcv","wc","rc"
    ],
    "Heart": [
        "age","sex","chest_pain","resting_bp","chol","fasting_bs",
        "resting_ecg","max_hr","exercise_angina","oldpeak","st_slope"
    ],
    # Liver: gender must be LAST per your request
    "Liver": [
        "age","total_bilirubin","direct_bilirubin","alkaline_phosphatase",
        "alanine_aminotransferase","aspartate_aminotransferase",
        "total_proteins","albumin","albumin_globulin_ratio","gender"
    ]
}

# ---------- Thresholds per your request ----------
THRESHOLDS = {
    "Diabetes": {"low": 0.40, "medium": 0.75},
    "Kidney": {"low": 0.40, "medium": 0.75},
    "Heart": {"low": 0.40, "medium": 0.65},
    "Liver": {"low": 0.40, "medium": 0.70}
}

# ---------- Precautions (low/medium/high) ----------
PRECAUTIONS = {
    "Diabetes": {
        "low": [
            "Maintain balanced diet and regular exercise.",
            "Avoid sugary drinks and processed food.",
            "Monitor glucose occasionally."
        ],
        "medium": [
            "Follow a physician-recommended meal plan.",
            "Monitor glucose more frequently and consult physician.",
            "Consider lifestyle changes and medical review."
        ],
        "high": [
            "High probability of diabetes — consult a diabetologist immediately.",
            "Begin strict glucose monitoring and urgent tests.",
            "Follow medical advice for medication and treatment."
        ]
    },
    "Kidney": {
        "low": [
            "Stay hydrated and avoid excess salt.",
            "Avoid unnecessary NSAIDs.",
            "Routine kidney function tests every 6–12 months."
        ],
        "medium": [
            "Adopt renal-friendly diet (moderate protein, low sodium).",
            "Monitor serum creatinine and urine tests regularly.",
            "Consult a nephrologist for early management."
        ],
        "high": [
            "High probability of CKD — urgent nephrology consultation required.",
            "Further tests (eGFR, ultrasound) and treatment planning needed.",
            "Avoid nephrotoxins and follow medical advice strictly."
        ]
    },
    "Heart": {
        "low": [
            "Maintain heart-healthy diet and 30 min exercise most days.",
            "Avoid smoking & excess alcohol.",
            "Monitor blood pressure periodically."
        ],
        "medium": [
            "Reduce salt and saturated fat intake.",
            "See a cardiologist; consider ECG or stress test.",
            "Avoid strenuous activity until cleared."
        ],
        "high": [
            "High probability of heart disease — seek immediate cardiology care.",
            "Urgent tests (ECG, troponin, echo) may be needed.",
            "Follow prescribed medications and interventions."
        ]
    },
    "Liver": {
        "low": [
            "Avoid alcohol and fatty/junk foods.",
            "Monitor liver function tests yearly.",
            "Stay hydrated and eat balanced diet."
        ],
        "medium": [
            "Stop alcohol completely; get LFT and ultrasound.",
            "Avoid hepatotoxic medications and supplements.",
            "Follow physician advice on diet and rest."
        ],
        "high": [
            "High probability of liver disease — urgent hepatology assessment required.",
            "Possible hospital-based investigations and treatments.",
            "Strict medical supervision and follow-up needed."
        ]
    }
}

# ---------- Helpers ----------
def get_model_and_scaler(disease):
    if disease == "Diabetes":
        return diabetes_model, diabetes_scaler
    if disease == "Kidney":
        return kidney_model, kidney_scaler
    if disease == "Heart":
        return heart_model, heart_scaler
    if disease == "Liver":
        return liver_model, liver_scaler
    raise ValueError("Unknown disease")

def map_risk(disease, prob):
    thr = THRESHOLDS[disease]
    if prob < thr["low"]:
        return "Low", PRECAUTIONS[disease]["low"]
    elif prob < thr["medium"]:
        return "Medium", PRECAUTIONS[disease]["medium"]
    else:
        return "High", PRECAUTIONS[disease]["high"]

# ---------- Routes ----------
@app.route("/")
def root():
    if "username" in session:
        return redirect(url_for("select_disease"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        uname = request.form.get("username","").strip()
        pwd = request.form.get("password","")
        if not uname or not pwd:
            flash("Provide username and password", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(username=uname).first():
            flash("Username already exists", "warning")
            return redirect(url_for("register"))
        user = User(username=uname, password_hash=generate_password_hash(pwd))
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username","").strip()
        pwd = request.form.get("password","")
        user = User.query.filter_by(username=uname).first()
        if user and check_password_hash(user.password_hash, pwd):
            session["username"] = uname
            flash("Login successful", "success")
            return redirect(url_for("select_disease"))
        flash("Invalid credentials", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("login"))

@app.route("/select", methods=["GET","POST"])
def select_disease():
    if "username" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        disease = request.form.get("disease")
        if disease not in FEATURES:
            flash("Select valid disease", "warning")
            return redirect(url_for("select_disease"))
        return redirect(url_for("enter_features", disease=disease))
    return render_template("select_disease.html", diseases=list(FEATURES.keys()))

@app.route("/enter/<disease>", methods=["GET"])
def enter_features(disease):
    if "username" not in session:
        return redirect(url_for("login"))
    
    # Ensure disease is valid
    if disease not in FEATURES:
        flash("Unknown disease", "danger")
        return redirect(url_for("select_disease"))
    
    # Get the list of feature names for this disease
    feature_names = FEATURES[disease]

    # Render template without examples or previous values
    return render_template(
        "enter_features.html",
        disease=disease,
        feature_names=feature_names
    )

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    if "username" not in session:
        return redirect(url_for("login"))
    if disease not in FEATURES:
        flash("Unknown disease", "danger")
        return redirect(url_for("select_disease"))

    # collect values in order defined in FEATURES[disease]
    values = []
    for fname in FEATURES[disease]:
        raw = request.form.get(fname, "").strip()
        try:
            v = float(raw)
        except Exception:
            v = 0.0
        values.append(v)

    X = np.array(values).reshape(1, -1)
    model, scaler = get_model_and_scaler(disease)
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(Xs)[0, -1])
    else:
        pred = model.predict(Xs)[0]
        prob = 1.0 if pred == 1 else 0.0

    risk, precautions = map_risk(disease, prob)

    # store history
    hist = History(
        username = session["username"],
        disease = disease,
        probability = prob,
        risk_level = risk,
        precautions = "||".join(precautions),
        inputs = "||".join([f"{k}={v}" for k,v in zip(FEATURES[disease], values)])
    )
    db.session.add(hist)
    db.session.commit()

    return render_template("result.html",
                           disease=disease,
                           probability=round(prob*100,2),
                           risk=risk,
                           precautions=precautions,
                           inputs=dict(zip(FEATURES[disease], values))
                           )

@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    rows = History.query.filter_by(username=session["username"]).order_by(History.created_at.desc()).all()
    return render_template("history.html", rows=rows)

# convenience route to go to select page
@app.route("/dashboard")
def dashboard():
    return redirect(url_for("select_disease"))

if __name__ == "__main__":
    app.run(debug=True)
