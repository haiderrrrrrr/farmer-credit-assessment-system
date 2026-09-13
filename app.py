#!/usr/bin/env python
"""
Flask front-end for the Farmer Credit Assessment System
* Uses PostgreSQL (works great with Python/psycopg2).
* Reads DB credentials + SECRET_KEY from .env
* Auto-creates `users` and `predictions` tables if they don't exist.
"""
import os
import glob
import json
import datetime
import traceback
import sys, os
import re
import smtplib
import io
import csv
import warnings
from pathlib import Path
from urllib.parse import urlparse
from email.mime.text import MIMEText
import subprocess
from flask import stream_with_context, send_file
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
import pandas as pd
from flask import (
    Flask, render_template, redirect, url_for,
    request, flash, Blueprint, g, jsonify,
    Response, abort
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from psycopg2 import errors as pg_errors
from dotenv import load_dotenv

RUNS_DIR = Path(__file__).parent / "runs"

# -- env & app setup ---------------------------------------------------------
load_dotenv()
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.urandom(32)

warnings.filterwarnings("ignore", message=".*serialized model.*", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)

# -- email configuration ------------------------------------------------------
GMAIL_USER       = os.getenv("GMAIL_USER")
GMAIL_PASS       = os.getenv("GMAIL_PASS")
CONTACT_RECEIVER = os.getenv("CONTACT_RECEIVER", GMAIL_USER)

login_manager = LoginManager(app)
login_manager.login_view = "auth.login"

# -- database helper ----------------------------------------------------------
def get_db():
    if "db" not in g:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            parsed = urlparse(database_url)
            g.db = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                dbname=parsed.path.lstrip("/"),
                cursor_factory=RealDictCursor
            )
        else:
            g.db = psycopg2.connect(
                host=os.getenv("PG_HOST", "localhost"),
                port=os.getenv("PG_PORT", "5432"),
                user=os.getenv("PG_USER", "postgres"),
                password=os.getenv("PG_PASS", ""),
                dbname=os.getenv("PG_DB", "farmer_credit"),
                cursor_factory=RealDictCursor
            )
        g.db.autocommit = False
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()

# -- initialize schema --------------------------------------------------------
def init_db():
    try:
        ddl_users = """
            CREATE TABLE IF NOT EXISTS users(
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                role TEXT   DEFAULT 'user'
            );
        """
        ddl_add_name = """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS name TEXT NOT NULL DEFAULT '';
        """
        ddl_pred = """
            CREATE TABLE IF NOT EXISTS predictions(
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id),
                ts      TIMESTAMPTZ,
                risk_level INT,
                risk_score DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                version TEXT,
                input_json JSONB
            );
        """
        db = get_db()
        with db.cursor() as cur:
            cur.execute(ddl_users)
            cur.execute(ddl_add_name)
            cur.execute(ddl_pred)
        db.commit()
        print("[OK] Database initialized successfully")
    except Exception as e:
        print(f"[WARN]  Database initialization failed: {e}")
        print("[NOTE] Please ensure PostgreSQL is running and update farmer_credit.env with correct credentials")
        print("[SETUP] You can also use SQLite by modifying the database configuration")

# Try to initialize database, but don't fail if it doesn't work
with app.app_context():
    try:
        init_db()
    except:
        pass

# -- Flask-Login user setup --------------------------------------------------
class User:
    def __init__(self, row):
        self.id      = row["id"]
        self.name    = row.get("name", "")
        self.email   = row["email"]
        self.pw_hash = row["pw_hash"]
        self.role    = row["role"]
    def is_authenticated(self): return True
    def is_active(self): return True
    def is_anonymous(self): return False
    def get_id(self): return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
    return User(row) if row else None

# -- auth blueprint ----------------------------------------------------------
auth_bp = Blueprint("auth", __name__, url_prefix="/")

@auth_bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw    = request.form["password"]
        db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
        if row and check_password_hash(row["pw_hash"], pw):
            user = User(row)
            login_user(user)
            # role-based redirect:
            if user.role == "admin":
                return redirect(url_for("admindashboard"))
            else:
                return redirect(url_for("dashboard"))
        flash("Invalid credentials")
    return render_template("auth/registration.html", show_signup=False)

@auth_bp.route("/signup", methods=["GET","POST"])
def signup():
    if request.method=="POST":
        name     = request.form["name"].strip()
        email    = request.form["email"].strip().lower()
        password = request.form["password"]
        if not name:
            flash("Name is required."); return render_template("auth/registration.html", show_signup=True)
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            flash("Invalid email.");    return render_template("auth/registration.html", show_signup=True)
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*\W).{8,}$', password):
            flash("Password must be >=8 chars."); return render_template("auth/registration.html", show_signup=True)
        pw_hash = generate_password_hash(password)
        db = get_db()
        try:
            with db.cursor() as cur:
                cur.execute("INSERT INTO users(name,email,pw_hash) VALUES (%s,%s,%s)", (name,email,pw_hash))
            db.commit(); flash("Account created - please sign in."); return redirect(url_for("auth.login"))
        except pg_errors.UniqueViolation:
            db.rollback(); flash("Email already registered.")
    return render_template("auth/registration.html", show_signup=True)

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("welcome"))

app.register_blueprint(auth_bp)

# -- predict endpoint --------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        print("[LOAD] API Predict called")
        print(f"[DATA] Model available: {MODEL is not None}")
        print(f"[DATA] Scaler available: {MODEL_SCALER is not None}")
        print(f"[DATA] Features available: {len(MODEL_FEATURES) if MODEL_FEATURES else 0}")
        
        if not MODEL or not MODEL_SCALER or not MODEL_FEATURES:
            return jsonify({"error": "Model not properly loaded"}), 500
        
        data = request.get_json(force=True)
        print(f"[INPUT] Received data: {data}")
        
        fields = [
            "Avg_Yield", "Total_Production", "Total_Area", "Rainfall_Volatility",
            "Production_Efficiency", "Fertilizer_Intensity", "Climate_Trend",
            "Price_Volatility", "Drought_Risk", "Soil_Quality", "Regional_Climate_Score",
            "Yield_Volatility", "Crop_Diversity", "Avg_Rainfall", "Temp_Volatility", "Soil_Fertility"
        ]
        sample = {f: float(data.get(f, 0)) for f in fields}
        print(f"[FEATURES] Sample data: {sample}")
        
        # Feature Engineering: Calculate derived features
        derived_features = {}
        
        # Basic calculations
        if sample['Total_Area'] > 0:
            derived_features['Yield_Rainfall_Ratio'] = sample['Avg_Yield'] / (sample['Avg_Rainfall'] + 1) * 1000
            derived_features['Avg_Productivity'] = sample['Total_Production'] / sample['Total_Area']
        else:
            derived_features['Yield_Rainfall_Ratio'] = 0.0
            derived_features['Avg_Productivity'] = 0.0
        
        # Risk calculations
        derived_features['Yield_Risk'] = (100 - sample['Production_Efficiency']) * 0.5 + sample['Yield_Volatility'] * 0.3
        derived_features['Production_Risk'] = sample['Price_Volatility'] * 0.4 + sample['Drought_Risk'] * 0.3
        derived_features['Climate_Risk'] = sample['Rainfall_Volatility'] * 0.4 + sample['Temp_Volatility'] * 0.3 + abs(sample['Climate_Trend']) * 0.3
        derived_features['Fertilizer_Risk'] = (100 - sample['Soil_Fertility']) * 0.4 + sample['Fertilizer_Intensity'] * 0.2
        
        # Trend calculations
        derived_features['Yield_Trend'] = sample['Climate_Trend'] * 0.5 + (100 - sample['Drought_Risk']) * 0.3
        derived_features['Production_Trend'] = sample['Production_Efficiency'] * 0.4 + (100 - sample['Price_Volatility']) * 0.3
        
        # Normalized scores
        derived_features['Drought_Risk_Normalized'] = sample['Drought_Risk'] * 0.8 + sample['Rainfall_Volatility'] * 0.2
        derived_features['Soil_Risk'] = (100 - sample['Soil_Quality']) * 0.6 + (100 - sample['Soil_Fertility']) * 0.4
        derived_features['Regional_Climate_Risk'] = sample['Regional_Climate_Score'] * 0.5 + derived_features['Climate_Risk'] * 0.5
        derived_features['Production_Efficiency_Risk'] = (100 - sample['Production_Efficiency']) * 0.7 + sample['Yield_Volatility'] * 0.3
        
        # Add random factor for realism (small variation)
        import random
        derived_features['Random_Factor'] = random.uniform(-5, 5)
        
        # Create final sample with only the features the model expects
        final_sample = {}
        
        # First, add all input features
        for field in fields:
            if field in MODEL_FEATURES:
                final_sample[field] = sample[field]
        
        # Add derived features that are in MODEL_FEATURES
        for feature, value in derived_features.items():
            if feature in MODEL_FEATURES:
                final_sample[feature] = value
        
        # Add any remaining MODEL_FEATURES with default values
        for feature in MODEL_FEATURES:
            if feature not in final_sample:
                final_sample[feature] = 0.0
        
        print(f"[FEATURES] Final sample: {final_sample}")
        print(f"[FEATURES] Total features: {len(final_sample)}")
        print(f"[FEATURES] Features in final sample: {list(final_sample.keys())}")
        
        # Verify all required features are present
        if len(final_sample) != len(MODEL_FEATURES):
            print(f"[ERROR] Feature count mismatch: final sample has {len(final_sample)}, model expects {len(MODEL_FEATURES)}")
            return jsonify({"error": "Feature count mismatch"}), 500
        
        # Create DataFrame with exact features the model expects
        X = pd.DataFrame([final_sample])[MODEL_FEATURES]
        print(f"[DATA] Input DataFrame shape: {X.shape}")
        print(f"[DATA] Input DataFrame columns: {list(X.columns)}")
        
        X_scaled = MODEL_SCALER.transform(X)
        print(f"[DATA] Scaled input shape: {X_scaled.shape}")
        
        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]
        confidence = max(probabilities) * 100
        
        print(f"[TARGET] Prediction: {prediction}")
        print(f"[DATA] Probabilities: {probabilities}")
        print(f"[CONFIDENCE] Confidence: {confidence}")
        
        risk_levels = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
        risk_level_name = risk_levels[prediction]
        
        print(f"[VERSION]  Risk level: {risk_level_name}")

        db = get_db()
        with db.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions(user_id,ts,risk_level,risk_score,confidence,version,input_json)
                VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING id
            """,(
                current_user.id,
                datetime.datetime.utcnow(),
                int(prediction),
                int(prediction),
                float(confidence),
                MODEL_VERSION,
                json.dumps(sample)
            ))
            pred_id = cur.fetchone()['id']
        db.commit()
        
        print(f"[SAVE] Saved to database with ID: {pred_id}")

        return jsonify({
            "id": pred_id,
            "risk_level": int(prediction),
            "risk_level_name": risk_level_name,
            "confidence": float(confidence),
            "probabilities": probabilities.tolist()
        })
        
    except Exception as e:
        print(f"[ERROR] Error in predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug-auth")
def debug_auth():
    """Debug endpoint to check authentication status"""
    try:
        if current_user and hasattr(current_user, 'is_authenticated'):
            if callable(current_user.is_authenticated):
                is_auth = current_user.is_authenticated()
            else:
                is_auth = current_user.is_authenticated
        else:
            is_auth = False
            
        return jsonify({
            "authenticated": is_auth,
            "user_id": current_user.id if current_user and is_auth else None,
            "user_email": current_user.email if current_user and is_auth else None,
            "session_active": True
        })
    except Exception as e:
        print(f"Debug auth error: {e}")
        return jsonify({
            "authenticated": False,
            "user_id": None,
            "user_email": None,
            "session_active": False,
            "error": str(e)
        })

@app.route("/api/test-model")
def test_model():
    """Test endpoint to verify model is working"""
    try:
        if not MODEL or not MODEL_SCALER or not MODEL_FEATURES:
            return jsonify({
                "status": "error",
                "message": "Model not loaded",
                "model_loaded": MODEL is not None,
                "scaler_loaded": MODEL_SCALER is not None,
                "features_loaded": len(MODEL_FEATURES) if MODEL_FEATURES else 0
            }), 500
        
        # Create a simple test sample with all features set to 0
        test_sample = {feature: 0.0 for feature in MODEL_FEATURES}
        X = pd.DataFrame([test_sample])[MODEL_FEATURES]
        X_scaled = MODEL_SCALER.transform(X)
        
        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]
        confidence = max(probabilities) * 100
        
        return jsonify({
            "status": "success",
            "message": "Model working correctly",
            "prediction": int(prediction),
            "confidence": float(confidence),
            "feature_count": len(MODEL_FEATURES),
            "model_type": str(type(MODEL)),
            "model_version": MODEL_VERSION
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": str(traceback.format_exc())
        }), 500

# -- train model endpoint ----------------------------------------------------
@app.route("/train-model")
@login_required
def train_model_page():
    if current_user.role != "admin":
        abort(403)
    return render_template("train_model.html")

@app.route("/train-model/stream")
@login_required
def train_model_stream():
    if current_user.role != "admin":
        abort(403)

    def generate():
        # locate the training script
        here   = os.path.dirname(__file__)
        script = os.path.join(here, "train_model.py")

        # launch the script under the same Python interpreter
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        # stream each line as an SSE 'data:' event
        for line in proc.stdout:
            yield f"data: {line.rstrip()}\n\n"

        proc.wait()

        # once finished, grab the newest model directory name
        latest = sorted(Path("Trained_models").glob("model_*"))[-1].name

        # signal completion, sending the run_id
        yield f"event: complete\ndata: {latest}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream"
    )

@app.route("/train-model/report/<run_id>")
@login_required
def train_model_report(run_id):
    if current_user.role != "admin":
        abort(403)
    path = Path("Trained_models") / run_id / f"training_report_{run_id}.html"
    return send_file(path)

# -- expert endpoint --------------------------------------------------------
@app.route("/api/expert/<int:pred_id>")
@login_required
def api_expert(pred_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT * FROM predictions WHERE id=%s",(pred_id,))
        row = cur.fetchone()
    if not row:
        return jsonify({"error":"Prediction not found"}),404

    input_json = row['input_json']
    risk_level = row['risk_level']
    confidence = row['confidence']
    ts         = row['ts']
    version    = row['version']

    X = pd.DataFrame([input_json])
    for feature in MODEL_FEATURES:
        if feature not in X.columns:
            X[feature] = 0.0
    X = X[MODEL_FEATURES]
    X_scaled = MODEL_SCALER.transform(X)

    # Feature importance analysis
    if hasattr(MODEL, 'feature_importances_'):
        feature_importances = MODEL.feature_importances_.tolist()
    else:
        feature_importances = [1.0/len(MODEL_FEATURES)] * len(MODEL_FEATURES)

    # Sensitivity analysis
    sensitivity = []
    for i, feature in enumerate(MODEL_FEATURES):
        X2 = X_scaled.copy()
        X2[0, i] = X2[0, i] * 1.5  # Increase feature by 50%
        new_pred = MODEL.predict(X2)[0]
        sensitivity.append({
            "feature": feature,
            "changed_risk": int(new_pred),
            "original_risk": risk_level
        })

    # Cohort analysis
    with db.cursor() as cur:
        cur.execute("SELECT risk_level FROM predictions")
        all_risks = [r['risk_level'] for r in cur.fetchall()]
    
    risk_counts = [all_risks.count(i) for i in range(5)]
    risk_names = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

    # Recommendations based on risk level
    recs = []
    if risk_level >= 4:
        recs.append("Immediate intervention required - high credit risk")
        recs.append("Consider alternative financing options")
        recs.append("Implement risk mitigation strategies")
    elif risk_level >= 3:
        recs.append("Monitor closely - moderate credit risk")
        recs.append("Consider reduced credit limits")
        recs.append("Implement regular monitoring")
    elif risk_level >= 2:
        recs.append("Standard credit assessment")
        recs.append("Regular monitoring recommended")
    else:
        recs.append("Low credit risk - favorable terms available")
        recs.append("Consider premium credit products")

    return jsonify({
        "feature_names": MODEL_FEATURES,
        "feature_importances": feature_importances,
        "sensitivity": sensitivity,
        "risk_counts": risk_counts,
        "risk_names": risk_names,
        "metrics": {
            "model_version": version,
            "generated_at": ts.isoformat()
        },
        "audit": {
            "input": input_json,
            "timestamp": ts.isoformat()
        },
        "recommendations": recs
    })

# -- download CSV endpoint ---------------------------------------------------
@app.route("/download-history")
@login_required
def download_history():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT ts, risk_level, confidence, version
              FROM predictions
             WHERE user_id=%s
             ORDER BY ts DESC
        """, (current_user.id,))
        rows = cur.fetchall()

    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Date & Time", "Risk Level", "Confidence (%)", "Model Version"])
    risk_names = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
    for r in rows:
        writer.writerow([
            r["ts"].strftime("%Y-%m-%d %H:%M"),
            risk_names[r['risk_level']],
            f"{r['confidence']:.1f}%",
            r["version"]
        ])
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition":"attachment;filename=credit_assessment_history.csv"}
    )

# -- load model --------------------------------------------------------------
def latest_model_path():
    model_files = glob.glob("Trained_models/latest_model.pkl")
    if model_files:
        print(f"[OK] Found model at: {model_files[0]}")
        return model_files[0]
    else:
        print("[WARN]  No trained model found. Please run train_model.py first.")
        return None

print("[LOAD] Loading model...")
MODEL_PATH = latest_model_path()
if MODEL_PATH:
    try:
        print(f"[FILE] Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            MODEL_DATA = pickle.load(f)
        
        print(f"[DATA] Model data keys: {list(MODEL_DATA.keys())}")
        
        MODEL = MODEL_DATA['best_model']
        MODEL_SCALER = MODEL_DATA['scalers']['features']
        MODEL_FEATURES = MODEL_DATA['feature_cols']
        MODEL_VERSION = MODEL_DATA['model_version']
        
        print(f"[OK] Model loaded successfully")
        print(f"[MODEL] Model type: {type(MODEL)}")
        print(f"[FEATURES] Features: {len(MODEL_FEATURES)}")
        print(f"[REPORT] Feature list: {MODEL_FEATURES}")
        print(f"[VERSION]  Version: {MODEL_VERSION}")
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        traceback.print_exc()
        MODEL = None
        MODEL_SCALER = None
        MODEL_FEATURES = []
        MODEL_VERSION = "No model available"
else:
    print("[ERROR] No model path available")
    MODEL = None
    MODEL_SCALER = None
    MODEL_FEATURES = []
    MODEL_VERSION = "No model available"

# -- core routes -------------------------------------------------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/dashboard", methods=["GET","POST"])
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admindashboard"))
    
    risk_level = None
    confidence = None
    if request.method=="POST":
        fields = [
            "Avg_Yield", "Total_Production", "Total_Area", "Rainfall_Volatility",
            "Production_Efficiency", "Fertilizer_Intensity", "Climate_Trend",
            "Price_Volatility", "Drought_Risk", "Soil_Quality", "Regional_Climate_Score"
        ]
        sample = {f: float(request.form.get(f, 0)) for f in fields}
        
        # Add missing features
        for feature in MODEL_FEATURES:
            if feature not in sample:
                sample[feature] = 0.0
        
        X = pd.DataFrame([sample])[MODEL_FEATURES]
        X_scaled = MODEL_SCALER.transform(X)
        
        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]
        confidence = max(probabilities) * 100
        
        db = get_db()
        with db.cursor() as cur:
                    cur.execute("""
            INSERT INTO predictions(user_id,ts,risk_level,risk_score,confidence,version,input_json)
            VALUES(%s,%s,%s,%s,%s,%s,%s)
        """, (
            current_user.id,
            datetime.datetime.utcnow(),
            int(prediction),
            int(prediction),
            float(confidence),
            MODEL_VERSION,
            json.dumps(sample)
        ))
        db.commit()
        risk_level = prediction

    model_runs = sorted(Path("Trained_models").glob("model_*"))
    latest_model = model_runs[-1].name if model_runs else Path(MODEL_PATH).name if MODEL_PATH else None
    return render_template("userDashboard.html",
                           risk_level=risk_level,
                           confidence=confidence,
                           model_version=MODEL_VERSION,
                           latest_model=latest_model)

@app.route("/overview")
@login_required
def overview():
    return render_template("overview.html")

@app.route("/admin/assessment-history")
@login_required
def admin_assessment_history():
    if current_user.role != "admin":
        abort(403)

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT
              p.ts,
              p.risk_level,
              p.confidence,
              p.version,
              u.name,
              u.email
            FROM predictions AS p
            JOIN users       AS u
              ON p.user_id = u.id
            ORDER BY p.ts DESC
        """)
        records = cur.fetchall()
    return render_template("admin-assessment-history.html", records=records)

@app.route("/download-history-all")
@login_required
def download_history_all():
    if current_user.role != "admin":
        abort(403)

    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
          SELECT
            u.name, u.email,
            p.ts, p.risk_level, p.confidence, p.version
          FROM predictions p
          JOIN users u ON p.user_id = u.id
          ORDER BY p.ts DESC
        """)
        rows = cur.fetchall()

    si = io.StringIO()
    writer = csv.writer(si)
    risk_names = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
    writer.writerow(["User", "Email", "Date & Time", "Risk Level", "Confidence (%)", "Model Version"])
    for r in rows:
        writer.writerow([
            r["name"] or "",
            r["email"],
            r["ts"].strftime("%Y-%m-%d %H:%M:%S"),
            risk_names[r['risk_level']],
            f"{r['confidence']:.1f}%",
            r["version"]
        ])

    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":"attachment;filename=all_assessment_history.csv"}
    )

@app.route("/predictor")
@login_required
def predictor():
    return render_template("predictor.html")

@app.route("/assessment-history")
@login_required
def assessment_history():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT id, ts, risk_level, confidence, version
              FROM predictions
             WHERE user_id=%s
             ORDER BY ts DESC
        """, (current_user.id,))
        records = cur.fetchall()
    return render_template("assessment-history.html", records=records)

@app.route("/result-details")
@login_required
def result_details():
    return render_template("result-details.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/admindashboard")
@login_required
def admindashboard():
    if current_user.role != "admin": 
        abort(403)
    return render_template("admindashboard.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/contact")
def contact():
    return render_template("contactUs.html")

@app.route("/send-contact-form", methods=["POST"])
def send_contact_form():
    data    = request.get_json(force=True) or {}
    name    = data.get("name","").strip()
    email   = data.get("email","").strip()
    phone   = data.get("phone","").strip()
    message = data.get("message","").strip()

    if not(name and email and phone and message):
        return jsonify({"error":"All fields required."}),400
    if not GMAIL_USER or not GMAIL_PASS:
        app.logger.error("Missing GMAIL creds")
        return jsonify({"error":"Email config missing"}),500

    html = f"""
      <h3>Contact Form Submission</h3>
      <p><strong>Name:</strong> {name}</p>
      <p><strong>Email:</strong> {email}</p>
      <p><strong>Phone:</strong> {phone}</p>
      <p><strong>Message:</strong></p>
      <p>{message}</p>
    """
    msg = MIMEText(html, "html")
    msg["Subject"]=f"Contact from {name}"
    msg["From"]=GMAIL_USER; msg["To"]=CONTACT_RECEIVER

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:
            smtp.login(GMAIL_USER,GMAIL_PASS)
            smtp.send_message(msg)
        return jsonify({"message":"Your message has been sent successfully!"}),200
    except Exception:
        app.logger.exception("Error sending contact email")
        return jsonify({"error":"Failed to send message; please try again later."}),500

# -- run ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


