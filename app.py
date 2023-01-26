
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('/content/trained_model.sav','rb'))

#loading the scaler object
loaded_std_scaler = pickle.load(open('/content/std_scaler.pkl','rb'))

def diabetes_prediction(input_data):

  
  input_data = np.asarray(input_data)

  #reshape array as it is single instance

  input_data_reshape = input_data.reshape(1,-1)

  x_test = loaded_std_scaler.transform(input_data_reshape)

  prediction = loaded_model.predict(x_test)
  y_pred = np.where(prediction >0.5,1,0)

  if (y_pred[0][0]==0):
      print('The person is not suffering from diabetes')
  else:
      print('The person is suffering from diabetes')

def main():

  #Giving title to web page
  st.title('Diabetes prediction web application')

  #getting the input from web page

  Glucose = st.text_input('Glucose Level')
  BloodPressure = st.text_input('BloodPressure Level')
  SkinThickness = st.text_input('SkinThickness')
  Insulin = st.text_input('Insulin Level')
  BMI = st.text_input('BMI Value')
  DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
  Age = st.text_input('Age')

  #code for the prediction
  diagnosis = ''

  #Creating button for prediction

  if st.button('Diabetes Test Result'):
      diagnosis = diabetes_prediction([Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

  st.success(diagnosis)

if __name__=='__main__':
  main()
