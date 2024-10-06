# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:29:14 2024

@author: lenovo
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/lenovo/OneDrive/Desktop/deploy ML-Model(API)/Dabetic-Prediction/trained_model.sav', 'rb'))


# creating the function for the prediction

def diabetic_prediction(input_data):
    
    input_data = (4,110,92,0,0,37.6,0.191,30)

    # changing the input data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
      return("Person is not diabetic")

    else:
       return("Person is   diabetic")
   
    
   
    
def main():
    
    
    # giving a title 
    
    st.title('Diabetes Prediction Web APP')
    # Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	
    
    # getting the  input dataset from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Number of Glucose')
    BloodPressure =  st.text_input('Number of BloodPressure')
    SkinThickness = st.text_input('Number of SkinThickness')
    Insulin       = st.text_input('Number of Insulin')
    BMI = st.text_input('Number of BMI')
    DiabetesPedigreeFunction = st.text_input('Value of DiabetesPedigreeFunction')
    Age = st.text_input('Number of Age')
    
    
    # code for prediction 
    
    diagnosis = ''
    
    # cewating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetic_prediction(([Pregnancies	,Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,	Age	]))
        
    
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    