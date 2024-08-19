import pandas as pd
import math
import pickle

# create a streamlit app

import streamlit as st

# create a title and a subheader
st.title('Depression Detection App')

# insert image at home page
st.image('depression_app.jpg', width=400)

# load the model
model = pickle.load(open('model.pkl', 'rb'))

# Function to use the loaded model to predict from a given list of features
def predict(X): 
    # Dictionary to hold the patient id for key 1 depressed, and 0 not depressed
    predictions = {0: [], 1: []}
    for row in X:
        patient_id = row[0]
        features = [row[1:]]
        y = model.predict(features)  # Predict for each patient features 
        predictions[y[0]].append(patient_id)  # Append the patient id and prediction to the list
    
    return predictions

def getInputs():
    # Add a file uploader to the sidebar for CSV input
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file, including headers
        input_data = pd.read_csv(uploaded_file)

        # Check if the file is not empty
        if not input_data.empty:
            # Convert all rows to a list of lists
            values_list = input_data.values.tolist()

            # If the Predict button is clicked, call the predict function
            if st.button('Predict'):
                predictions = predict(values_list)  # Pass the list of lists to the predict function
                return predictions
        else:
            st.write("The uploaded file is empty. Please upload a valid CSV file.")
    else:
        st.write("Please upload a CSV file using the button on the left side")


if __name__ == "__main__":
    y = getInputs()
    
    if y is not None:

        if len(y[1]) > 0:
            # Go through list of patient ID for predicted depressed
            st.header('Detected Risk of Depression')
            st.image('2depression.png', width=200)   
            y[1].sort()
            for i in y[1]:
                st.write(f'Patient ID:\t {i}')

        if len(y[0]) > 0:
            # Go through list of patient ID for predicted not depressed
            st.header('Detected Not Depressed')
            st.image('1undepression.png', width=200)
            y[0].sort()
            for i in y[0]:
                st.write(f'Patient ID:\t {i}')  