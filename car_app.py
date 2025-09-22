import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and data
model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('Cleaned Car.csv')

st.title("🚗 Car Price Predictor")

# Dropdown for company
companies = sorted(car['company'].unique())
company = st.selectbox("Select Company", companies)

# Filter car models based on selected company
filtered_models = car[car['company'] == company]['name'].unique()
car_model = st.selectbox("Select Car Model", sorted(filtered_models))

# Other inputs
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()
year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
kms_driven = st.number_input("KMs Driven", min_value=0, max_value=500000, step=500)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹ {np.round(prediction[0], 2)}")
