from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import json
import os

# --- App Setup ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_security_123'  # Important for session management
USER_FILE = 'users.json'

# --- JSON Helper Functions ---
def read_users():
    """Reads the list of users from the JSON file."""
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, 'w') as f: json.dump([], f)
        return []
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def write_users(users):
    """Writes the list of users to the JSON file."""
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# --- Load All Simplified Models ---
try:
    diabetes_model = pickle.load(open('models/diabetes_model_simple.pkl', 'rb'))
    breast_cancer_model = pickle.load(open('models/breast_cancer_model_simple.pkl', 'rb'))
    heart_model = pickle.load(open('models/heart_model_simple.pkl', 'rb'))
    liver_model = pickle.load(open('models/liver_model_simple.pkl', 'rb'))
    print("✅ All simplified models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: Model file not found. Ensure all 'train_..._simple.py' scripts have been run. Details: {e}")
except Exception as e:
    print(f"❌ An error occurred while loading models: {e}")

# --- Page Rendering & Authentication Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = read_users()
        user = next((u for u in users if u['username'] == username and u['password'] == password), None)
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = read_users()
        if any(u['username'] == username for u in users):
            return render_template('register.html', error="Username already exists")
        users.append({'username': username, 'password': password})
        write_users(users)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

# --- Prediction Form Routes ---
@app.route('/diabetes')
def diabetes_page():
    if 'username' in session: return render_template('predict_diabetes_simple.html')
    return redirect(url_for('login'))

@app.route('/breast_cancer')
def breast_cancer_page():
    if 'username' in session: return render_template('predict_breast_cancer_simple.html')
    return redirect(url_for('login'))

@app.route('/heart_disease')
def heart_disease_page():
    if 'username' in session: return render_template('predict_heart_disease_simple.html')
    return redirect(url_for('login'))

@app.route('/liver_disease')
def liver_disease_page():
    if 'username' in session: return render_template('predict_liver_disease_simple.html')
    return redirect(url_for('login'))

# --- Prediction Logic Routes with Probability ---
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if 'username' not in session: return redirect(url_for('login'))
    glucose = int(request.form.get('blood_glucose_level')); bmi = float(request.form.get('bmi')); hba1c = float(request.form.get('HbA1c_level')); age = int(request.form.get('age')); hyper = int(request.form.get('hypertension'))
    features = np.array([[glucose, bmi, hba1c, age, hyper]])
    prediction = diabetes_model.predict(features)
    probabilities = diabetes_model.predict_proba(features)
    high_risk_prob = round(probabilities[0][1] * 100, 2)
    
    result = {'disease': 'Diabetes', 'prediction_value': int(prediction[0]), 'probability': high_risk_prob}
    if prediction[0] == 1:
        result['outcome'] = f'High Risk ({high_risk_prob}%)'; result['suggestions'] = ["Consult a healthcare professional.", "Monitor blood sugar levels."]
    else:
        result['outcome'] = f'Low Risk ({100 - high_risk_prob:.2f}%)'; result['suggestions'] = ["Maintain a healthy lifestyle.", "Continue with regular exercise."]
    return render_template('results.html', result=result)

@app.route('/predict/breast_cancer', methods=['POST'])
def predict_breast_cancer():
    if 'username' not in session: return redirect(url_for('login'))
    cp_worst = float(request.form.get('concave points_worst')); perim_worst = float(request.form.get('perimeter_worst')); rad_worst = float(request.form.get('radius_worst')); area_worst = float(request.form.get('area_worst')); concav_worst = float(request.form.get('concavity_worst'))
    features = np.array([[cp_worst, perim_worst, rad_worst, area_worst, concav_worst]])
    prediction = breast_cancer_model.predict(features)
    probabilities = breast_cancer_model.predict_proba(features)
    high_risk_prob = round(probabilities[0][1] * 100, 2)
    
    result = {'disease': 'Breast Cancer', 'prediction_value': int(prediction[0]), 'probability': high_risk_prob}
    if prediction[0] == 1:
        result['outcome'] = f'High Risk ({high_risk_prob}%)'; result['suggestions'] = ["Immediate consultation with an oncologist is strongly recommended."]
    else:
        result['outcome'] = f'Low Risk ({100 - high_risk_prob:.2f}%)'; result['suggestions'] = ["Continue with regular self-examinations and health check-ups."]
    return render_template('results.html', result=result)

@app.route('/predict/heart_disease', methods=['POST'])
def predict_heart_disease():
    if 'username' not in session: return redirect(url_for('login'))
    cp = int(request.form.get('cp')); thalach = int(request.form.get('thalach')); ca = int(request.form.get('ca')); oldpeak = float(request.form.get('oldpeak')); thal = int(request.form.get('thal'))
    features = np.array([[cp, thalach, ca, oldpeak, thal]])
    prediction = heart_model.predict(features)
    probabilities = heart_model.predict_proba(features)
    high_risk_prob = round(probabilities[0][1] * 100, 2)
    
    result = {'disease': 'Heart Disease', 'prediction_value': int(prediction[0]), 'probability': high_risk_prob}
    if prediction[0] == 1:
        result['outcome'] = f'High Risk ({high_risk_prob}%)'; result['suggestions'] = ["Seek immediate consultation with a cardiologist.", "A healthy diet is crucial."]
    else:
        result['outcome'] = f'Low Risk ({100 - high_risk_prob:.2f}%)'; result['suggestions'] = ["Maintain a heart-healthy diet and active lifestyle.", "Monitor blood pressure."]
    return render_template('results.html', result=result)

@app.route('/predict/liver_disease', methods=['POST'])
def predict_liver_disease():
    if 'username' not in session: return redirect(url_for('login'))
    alkphos = int(request.form.get('alkphos')); sgot = int(request.form.get('sgot')); total_bilirubin = float(request.form.get('total_bilirubin')); age = int(request.form.get('age')); albumin = float(request.form.get('albumin'))
    features = np.array([[alkphos, sgot, total_bilirubin, age, albumin]])
    prediction = liver_model.predict(features)
    probabilities = liver_model.predict_proba(features)
    high_risk_prob = round(probabilities[0][1] * 100, 2)
    
    result = {'disease': 'Liver Disease', 'prediction_value': int(prediction[0]), 'probability': high_risk_prob}
    if prediction[0] == 1:
        result['outcome'] = f'High Risk ({high_risk_prob}%)'; result['suggestions'] = ["Consult a hepatologist promptly.", "Avoid alcohol."]
    else:
        result['outcome'] = f'Low Risk ({100 - high_risk_prob:.2f}%)'; result['suggestions'] = ["Maintain a healthy diet.", "Limit alcohol consumption."]
    return render_template('results.html', result=result)

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)