import pandas as pd
from PIL import Image
import pickle
import sklearn
import streamlit as st
my_model = pickle.load(open("rf_model.sav", "rb"))
st.title("BANK ACCOUNT PREDICTION APP")
img = Image.open("pexels-anna-shvets-4482900.jpg")
st.image(img, width=350)
def user_report():
    country = st.sidebar.slider('country', 0.0, 4.0, 1.0)
    cellphone_access = st.sidebar.slider('cellphone_access', 0.0, 2.0, 1.0)
    education_level = st.sidebar.slider('education_level', 0.0, 8.0, 1.0)
    job_type = st.sidebar.slider('job_type', 0.0, 12.0, 0.1)

    input_data = {
        'country': country,
        'cellphone_access': cellphone_access,
        'education_level': education_level,
        'job_type': job_type
    }
    data = pd.DataFrame(input_data, index=[0])
    return data

user_data = user_report()
st.write(user_data)
prediction = my_model.predict(user_data)
if (prediction==0):
    st.success('Yes')
else:
    st.success('No')

