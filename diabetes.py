import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🚀 Starting SIMPLIFIED Diabetes Model Training (Corrected)...")

# 1. Load the dataset
# Use the actual name of your CSV file here
df = pd.read_csv('dataset/diabetes.csv') 

# 2. Select the top 5 features available in your file
top_5_features = [
    'blood_glucose_level',
    'bmi',
    'HbA1c_level',
    'age',
    'hypertension'
]
# The target column is 'diabetes'
target_column = 'diabetes'

X = df[top_5_features]
y = df[target_column]

print(f"✅ Training with features: {top_5_features}")

# 3. Split, Train, and Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"📈 Simplified Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# 4. Save the new, simplified model
with open('models/diabetes_model_simple.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Simplified model saved as 'models/diabetes_model_simple.pkl'")