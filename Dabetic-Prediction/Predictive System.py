# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:59:00 2024

@author: lenovo
"""

import numpy as np
import pickle 

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/lenovo/OneDrive/Desktop/deploy ML-Model(API)/Dabetic-Prediction/trained_model.sav', 'rb'))



input_data = (4,110,92,0,0,37.6,0.191,30)

# changing the input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
  print("Person is not diabetic")

else:
    print("Person is   diabetic")