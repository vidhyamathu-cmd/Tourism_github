import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="vidhyamathu/Tourism-Prediction", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Purchase Predictor
st.title("Wellness Tourism Purchase Predictor")
st.write("""
An interactive UI designed to analyze customer details and generate purchase predictions.
Please enter the relevent data below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Gender = st.selectbox("Gender", ["Female", "Male"])
Age = st.number_input("Age", min_value=1, max_value=100, value=20, step=1)
CityTier = st.number_input("CityTier", min_value=0, max_value=3, value=3, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, max_value=300, value=50)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer","Small Business","Large Business"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=300, value=10)
ProductPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic","Standard","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=0, max_value=5, value=3)
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Divorced","Married","Unmarried"])
NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=30, value=2, step=1)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=1, step=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=0, max_value=5, value=1, step=1)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=1, value=1, step=1)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=1, step=1)
Designation = st.selectbox("Designation", ["Manager", "Executive","Senior Manager","Manager","AVP"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=100, max_value=1000000, value=500, step=100)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Gender': Gender,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome


}])


if st.button("Tourism Purchase Predict"):
    prediction = model.predict(input_data)[0]
    result = "Purachase success" if prediction == 1 else "Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
