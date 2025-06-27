import pandas as pd
import pickle as pk
import streamlit as st

# Load the trained model (LinearRegression pipeline)
model = pk.load(open(r'C:\Users\JOYDEEP PAUL\Desktop\ml_dl\house_price_model.pkl', 'rb'))

# App Header
st.header('🏡 House Price Prediction App')

# Load cleaned data for location choices
data = pd.read_csv(r'C:\Users\JOYDEEP PAUL\Desktop\ml_dl\cleaned_house_price_data.csv')

# User Inputs
loc = st.selectbox('Choose location', sorted(data['location'].unique().tolist()))
sqft = st.number_input('Enter the area of the house (in square feet)', min_value=100.0, step=50.0)
beds = st.number_input('Number of bedrooms', min_value=1, max_value=10, value=2)
baths = st.number_input('Number of bathrooms', min_value=1, max_value=10, value=2)
balconies = st.number_input('Number of balconies', min_value=0, max_value=10, value=1)

# Predict button
if st.button('Predict Price'):

    # Prevent division by zero
    if beds == 0:
        st.error("Number of bedrooms cannot be zero.")
    else:
        # Prepare input
        input_df = pd.DataFrame([[loc, sqft, baths, balconies, beds]],columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])


        # Add derived feature (used in model training)
        input_df['sqft_per_bed'] = input_df['total_sqft'] / input_df['bedrooms']

        # Reorder columns to match model training
        input_df = input_df[['location', 'total_sqft', 'bath', 'balcony', 'bedrooms', 'sqft_per_bed']]

        # Debug: show input (optional)
        st.write("📥 Model Input:", input_df)

        # Make prediction
        try:
            predicted_price = model.predict(input_df)[0]  # Already in ₹ lakhs
            predicted_price = max(0, predicted_price)     # Avoid negative predictions

            # Format nicely
            st.success(f"✅ Estimated House Price: ₹ {round(predicted_price * 1_00_000):,}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
