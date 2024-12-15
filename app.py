import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')  

# Map predicted classes to meaningful names
class_names = {
    0: "Urban Area with High Water Access",
    1: "Rural Areas with Poor Water Access",
    2: "Semi-Urban/Transitional Area",
    3: "Rural Areas with Water Challenges"
}

# Define the feature names and their user-friendly labels
features = {
    'pop_u': 'Urban Population Percentage',
    'pop_r': 'Rural Population Percentage',
    'wat_bas_n': 'Basic Water Services (National)',
    'wat_bas_r': 'Basic Water Services (Rural)',
    'wat_bas_u': 'Basic Water Services (Urban)',
    'wat_lim_n': 'Limited Water Services (National)',
    'wat_sur_n': 'Surface Water (National)',
    'wat_sur_r': 'Surface Water (Rural)',
    'wat_unimp_r': 'Unimproved Water (Rural)',
    'wat_lim_u': 'Limited Water Services (Urban)',
    'wat_lim_r': 'Limited Water Services (Rural)'
}

def main():
    # Add a background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://i.postimg.cc/mZPcNxvV/download-2.jpg');
            background-size: cover;
            background-position: center;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Water Quality Classification in Country")
    st.write("Provide the relevant metrics to classify water quality.")

    # Collect user inputs for each feature
    user_input = []
    for feature, label in features.items():
        value = st.number_input(label, min_value=0.0, max_value=100.0, step=0.1)
        user_input.append(value)

    # Predict water quality when the user clicks the button
    if st.button("Classify Water Quality"):
        # Convert input to numpy array and scale it
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)  # Use the pre-trained scaler

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        predicted_class = class_names.get(prediction, "Unknown")

        # Display the result
        st.success(f"Predicted Water Quality: {predicted_class}")

if __name__ == "__main__":
    main()
