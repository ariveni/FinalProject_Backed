from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from collections import Counter
#from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)
# Load the models
final_rf_model = pickle.load(open("randomforest.pkl", "rb"))
final_nb_model = pickle.load(open("GaussianNB.pkl", "rb"))
final_svm_model = pickle.load(open("svm_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl","rb"))

@app.route("/predict", methods=['POST'])
def predi():
    symptoms = request.json.get('symptoms', [])  # assuming symptoms are passed as JSON array
    # List of symptoms
    symptoms_list = np.array(['itching', 'skin_rash', 'nodal_skin_eruptions',
       'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
       'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
       'vomiting', 'burning_micturition', 'spotting_ urination',
       'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
       'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
       'patches_in_throat', 'irregular_sugar_level', 'cough',
       'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'indigestion', 'headache', 'yellowish_skin',
       'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
       'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
       'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
       'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
       'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'pain_during_bowel_movements',
       'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
       'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
       'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
       'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
       'excessive_hunger', 'extra_marital_contacts',
       'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
       'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'spinning_movements',
       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
       'altered_sensorium', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic _patches',
       'watering_from_eyes', 'increased_appetite', 'polyuria',
       'family_history', 'mucoid_sputum', 'rusty_sputum',
       'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'receiving_unsterile_injections',
       'coma', 'stomach_bleeding', 'distention_of_abdomen',
       'history_of_alcohol_consumption', 'fluid_overload.1',
       'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
       'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
       'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
       'inflammatory_nails', 'blister', 'red_sore_around_nose',
       'yellow_crust_ooze'])

    symptom_index = {}
    for index, value in enumerate(symptoms_list):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
        "symptom_index": symptom_index,
        "predictions_classes": encoder.classes_
    }

    def predictDisease(symptoms):
        print(symptoms)
        #ymptoms = symptoms.split(",")
        print(symptoms)

        # Creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

        # Reshaping the input data and converting it
        # into a suitable format for model predictions
        input_data = np.array(input_data).reshape(1, -1)

        # Generating individual outputs
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

        # Finding the mode using Counter
        predictions_list = [rf_prediction, nb_prediction, svm_prediction]
        count_predictions = Counter(predictions_list)
        final_prediction = count_predictions.most_common(1)[0][0]

        predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
        }
        return predictions

    #result = predictDisease(symptoms_list, final_rf_model, final_nb_model, final_svm_model) $env:FLASK_APP = "app1.py"
    result = predictDisease(symptoms)
    data = result["final_prediction"] 
    # spliting function 
    def split_data(data):
        disease_name, drug_data = data.split(',',1)
        disease_name = disease_name.strip()
        drugs = [drug.strip() for drug in drug_data.strip('{}').split(',')]
        return disease_name, drugs
    print(data)
    disease,drugs = split_data(data)
    drugs[0]= drugs[0].replace("Drug={","")

    print(result["final_prediction"])
    return jsonify({
        "disease": disease,
        "prescription" : drugs
    })


if __name__ == '__main__':
    app.run(debug=True, port=9090)