import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🚀 Starting SIMPLIFIED Liver Disease Model Training...")

# 1. Load the dataset
try:
    df = pd.read_csv('dataset/liver.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'dataset/liver.csv' not found. Please check the path and filename.")
    exit()

# 2. Preprocess the Data
# Fills any missing numeric values with that column's average
df.fillna(df.mean(numeric_only=True), inplace=True)
# Converts the target column to 1s and 0s
df['Result'] = df['Result'].map({1: 1, 2: 0})
print("✅ Data preprocessing complete.")


# 3. Select the top 5 features USING YOUR EXACT PROVIDED NAMES
top_5_features = [
    'Alkphos Alkaline Phosphotase',
    'Sgot Aspartate Aminotransferase',
    'Total Bilirubin',
    'Age of the patient',
    'ALB Albumin'
]
target_column = 'Result'

# 4. Attempt to train the model with these names
try:
    X = df[top_5_features]
    y = df[target_column]
    print(f"✅ Training with features: {top_5_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"📈 Simplified Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    # Save the model
    with open('models/liver_model_simple.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Simplified model saved as 'models/liver_model_simple.pkl'")
    print("🎉 Liver disease training process finished.")

except KeyError as e:
    print("\n" + "="*60)
    print("❌ FATAL KeyError: The script failed as predicted.")
    print("   This confirms that one of the names in the 'top_5_features' list")
    print("   does NOT EXACTLY match a column header in your 'liver.csv' file.")
    print(f"   The name(s) Python could not find: {e}")
    print("="*60 + "\n")
    print("   Please use the 'check_columns.py' script from our previous conversation")
    print("   to find the real, exact names from your file.")
    exit()