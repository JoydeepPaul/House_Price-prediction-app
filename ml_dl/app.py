import pandas as pd
import pickle as pk
import streamlit as st

# Load the trained model
model = pk.load(open(r'C:\Users\JOYDEEP PAUL\Desktop\ml_dl\house_price_model.pkl', 'rb'))

# App Header
st.header('House Price Prediction App')

# Load cleaned dataset to get location options
data = pd.read_csv(r'C:\Users\JOYDEEP PAUL\Desktop\ml_dl\cleaned_house_price_data.csv')

# Input fields from user
loc = st.selectbox('Choose location', data['location'].unique().tolist())
sqft = st.number_input('Enter the area of the house in square feet', min_value=100.0, step=50.0)
beds = st.number_input('Enter the number of bedrooms', min_value=1, max_value=10, value=1)
baths = st.number_input('Enter the number of bathrooms', min_value=1, max_value=10, value=1)
balconies = st.number_input('Enter the number of balconies', min_value=0, max_value=10, value=0)

# Prepare input DataFrame
input = pd.DataFrame([[loc, sqft, beds, baths, balconies]],columns=['location', 'total_sqft', 'bedrooms', 'bath', 'balcony'])

# Predict button
if st.button('Predict Price'):
    # Add derived feature required by model
    input['sqft_per_bed'] = input['total_sqft'] / input['bedrooms']

    # Optional: show the input for debugging
    st.write("Input to model:", input)

    # Predict and ensure non-negative output
    raw_prediction = model.predict(input)[0]
    prediction = max(0, raw_prediction)

    # Display output
    out_str = 'Price of the house is: ₹' + str(round(prediction * 100000, 2))
    st.success(out_str)
