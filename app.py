from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load("model/student_model.pkl")

def load_data(file_path="data/StudentsPerformance.csv"):
    return pd.read_csv(file_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                return redirect(url_for('dashboard', filename=file.filename))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    filename = request.args.get('filename', default="StudentsPerformance.csv")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_data(file_path)

    # Average by Gender
    gender_avg = df.groupby('gender')[['math score', 'reading score', 'writing score']].mean().reset_index()
    gender_data = gender_avg.to_dict(orient='list')

    # Average by Race
    race_avg = df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].mean().reset_index()
    race_data = race_avg.to_dict(orient='list')

    return render_template('dashboard.html',
                           gender_data=json.dumps(gender_data),
                           race_data=json.dumps(race_data))

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    race = int(request.form['race'])
    lunch = int(request.form['lunch'])
    prep = int(request.form['prep'])
    math = float(request.form['math'])
    reading = float(request.form['reading'])
    writing = float(request.form['writing'])

    features = np.array([[gender, race, lunch, prep, math, reading, writing]])
    prediction = model.predict(features)[0]
    result = "High Performer" if prediction == 1 else "Low Performer"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
