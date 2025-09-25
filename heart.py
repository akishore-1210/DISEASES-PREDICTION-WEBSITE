import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🚀 Starting SIMPLIFIED Heart Disease Model Training...")

# 1. Load the dataset
df = pd.read_csv('dataset/heart.csv')

# 2. Select the top 5 most important features
top_5_features = [
    'cp',
    'thalach',
    'ca',
    'oldpeak',
    'thal'
]

X = df[top_5_features]
y = df['target']

print(f"✅ Training with features: {top_5_features}")

# 3. Split, Train, and Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"📈 Simplified Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# 4. Save the new, simplified model
with open('models/heart_model_simple.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Simplified model saved as 'models/heart_model_simple.pkl'")