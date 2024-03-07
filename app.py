import csv
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        top_row = next(reader)
        return top_row
    
model = joblib.load('./model/random_forest.pkl')
all_symptoms = read_csv('./datasets/Training.csv')
all_symptoms = all_symptoms[:-1]

def encode_symptoms(symptoms):
    encoded_symptoms = np.zeros(len(all_symptoms))
    for symptom in symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            encoded_symptoms[index] = 1
    return encoded_symptoms

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = [x for x in request.form.values()]
        symptoms_encoded = encode_symptoms(symptoms)
        prediction = model.predict([symptoms_encoded])
        return render_template('index.html', symptoms=all_symptoms, prediction_text='The patient has {}'.format(prediction[0]))
    return render_template('index.html', symptoms=all_symptoms)

if __name__ == '__main__':
    app.run(debug=True)
