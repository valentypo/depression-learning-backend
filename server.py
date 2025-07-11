from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, origins=["https://machine-learning-mocha.vercel.app"])

try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", str(e))

@app.route("/")
def home():
    return "Depression Learning Backend is running."

@app.route('/api/predict', methods=['POST'])
def predict():
    rename_map = {
        "age": "Age",
        "profession": "Profession",
        "cgpa": "CGPA",
        "dietaryHabits": "Dietary Habits",
        "sleepDuration": "Sleep Duration",
        "academicPressure": "Academic Pressure",
        "studySatisfaction": "Study Satisfaction",
        "degree": "Degree",
        "suicidalThoughts": "Have you ever had suicidal thoughts ?",
        "familyHistory": "Family History of Mental Illness",
        "studyHours": "Study Hours",
        "financialStress": "Financial Stress",
        "gender": "Gender"
    }
    data = request.get_json()

    df = pd.DataFrame([data])

    df['degree'] = df['degree'].apply(lambda x: "Class 12" if x == "High School and lower" else x)
    
    if (df['degree'] == "Class 12").any():
        df = df.drop(["degreeName", "degreeType"], axis=1)
    elif (df['degree'] == "Bachelor").any() or (df['degree'] == "Master").any() or (df['degree'] == "Doctorate").any():
        if (df['degreeType'] == "Others").any():
            df = df.drop(["degreeName"], axis=1)
            df['degree'] = df['degreeType'].apply(lambda x: "Others" if x == "Others" else x)
            df = df.drop(["degreeType"], axis=1)
        else:
            df['degree'] = df['degreeName']
            df = df.drop(["degreeName", "degreeType"], axis=1)

    df = df.rename(columns=rename_map)

    result = model.predict_proba(df)[0][1]

    print('Received prediction request:', df)
    return jsonify({'prediction': result.item()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', debug=True, port=port)