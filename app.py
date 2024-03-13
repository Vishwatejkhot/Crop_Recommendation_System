import streamlit as st
import pickle
import numpy as np

# Load the model
with open('Data/my_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Define a dictionary to map numeric labels to crop names
label_to_crop = {
    0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas',
    5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate',
    10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon',
    15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
    20: 'jute', 21: 'coffee'
}

# Define a function to make predictions
def predict(input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = classifier.predict(input_array)
    # Map numeric prediction to crop name
    predicted_crop = label_to_crop.get(prediction[0], "Unknown")
    return predicted_crop

# Streamlit UI
st.title('Crop Prediction Demo')

# Sidebar with input fields
st.sidebar.header('Input Parameters')

# Add input fields for parameters
N = st.sidebar.number_input('Nitrogen (N)', value=0.0)
P = st.sidebar.number_input('Phosphorus (P)', value=0.0)
K = st.sidebar.number_input('Potassium (K)', value=0.0)
temperature = st.sidebar.number_input('Temperature', value=0.0)
humidity = st.sidebar.number_input('Humidity', value=0.0)
ph = st.sidebar.number_input('pH', value=0.0)
rainfall = st.sidebar.number_input('Rainfall', value=0.0)

# Button to trigger prediction
if st.sidebar.button('Predict'):
    # Call the predict function
    prediction = predict([N, P, K, temperature, humidity, ph, rainfall])
    st.write('Predicted Crop:', prediction)

