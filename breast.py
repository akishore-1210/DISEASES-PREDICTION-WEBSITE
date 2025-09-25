import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🚀 Starting SIMPLIFIED Breast Cancer Model Training...")

# 1. Load the dataset
df = pd.read_csv('dataset/breastcancer.csv')

# 2. Preprocess the target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 3. Select the top 5 most important features
top_5_features = [
    'concave points_worst',
    'perimeter_worst',
    'radius_worst',
    'area_worst',
    'concavity_worst'
]

X = df[top_5_features]
y = df['diagnosis']

print(f"✅ Training with features: {top_5_features}")

# 4. Split, Train, and Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"📈 Simplified Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# 5. Save the new, simplified model
with open('models/breast_cancer_model_simple.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Simplified model saved as 'models/breast_cancer_model_simple.pkl'")