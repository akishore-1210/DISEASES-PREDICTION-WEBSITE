Health Predictor Pro - AI-Powered Disease Prediction System
Health Predictor Pro is a sophisticated web application that leverages machine learning to predict the risk of four major diseases: Diabetes, Breast Cancer, Heart Disease, and Liver Disease. It features a secure user authentication system, an elegant and modern user interface, and provides insightful, graphical results to help users understand their health risks.

✨ Features
Four Machine Learning Models: Utilizes trained scikit-learn models to predict the risk for four different diseases based on key health metrics.

Secure User Authentication: Users can register and log in to a personal dashboard. User data is managed securely using a JSON file.

Attractive & Modern UI: A visually appealing, eye-catching interface with a dynamic background image and "glassmorphism" card design for a professional user experience.

Graphical Risk Representation: The prediction results are displayed with a dynamic "Risk Gauge" chart, providing an immediate and easy-to-understand visual summary of the user's risk level.

Prediction Probability: Instead of a simple binary outcome, the application shows the model's confidence as a percentage (e.g., "High Risk (82%)"), giving users more detailed insight.

Actionable Suggestions: Provides clear, actionable health suggestions based on the prediction outcome.

🛠️ Technology Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NumPy

Frontend: HTML, CSS, Bootstrap 5

Data Visualization: Chart.js

User Management: JSON

📂 Project Structure
disease_prediction_project/
│
├── 📁 dataset/
│   ├── diabetes.csv
│   ├── breastcancer.csv
│   └── ... (and other datasets)
│
├── 📁 models/
│   ├── diabetes_model_simple.pkl
│   └── ... (and other trained .pkl models)
│
├── 📁 templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── results.html
│   └── ... (all prediction form HTML files)
│
├── 🐍 app.py                  # Main Flask application
├── 🐍 train_diabetes_simple.py # Scripts to train the ML models
├── 📄 users.json               # Stores user credentials
└── 📄 README.md                 # You are here!

🚀 Setup and Installation
Follow these steps to get the project running on your local machine.

Prerequisites
Python 3.7+

pip (Python package installer)

Installation Steps
Clone the repository (or set up the project folder):

git clone [https://your-repository-url.com/](https://your-repository-url.com/)
cd disease_prediction_project

Create and activate a virtual environment:

Windows:

python -m venv venv
.\venv\Scripts\activate

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

Install the required libraries:
Create a requirements.txt file with the following content:

Flask
scikit-learn
pandas
numpy

Then, install them using pip:

pip install -r requirements.txt

▶️ How to Run the Application
1. Train the Machine Learning Models
Before running the web server, you must train the simplified models. Run each of the training scripts from your terminal:

python train_diabetes_simple.py
python train_breast_cancer_simple.py
python train_heart_disease_simple.py
python train_liver_disease_simple.py

This will create the necessary .pkl files in your models/ folder.

2. Run the Web Server
Once the models are trained, start the Flask application:

python app.py

The application will now be running at http://127.0.0.1:5000.

📖 How to Use the Application
Open your web browser and navigate to http://127.0.0.1:5000.

Register for a new account or Login if you already have one.

From your Dashboard, select which disease you would like to predict.

Fill in the 5 required health metrics on the prediction form.

Click the "Predict" button.

View your result, which includes the risk probability, a visual gauge chart, and personalized health suggestions.

🔮 Future Enhancements
This project has a solid foundation and can be extended with more advanced features, such as:

PDF Report Generation: Allow users to download a professional PDF of their results.

Prediction History: Save a user's prediction history to a database so they can track their health over time.

Explainable AI (XAI): Show which specific factors contributed most to a user's prediction.