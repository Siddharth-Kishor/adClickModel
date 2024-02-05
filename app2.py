import streamlit as st
import numpy as np
import pickle

with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

def convertedData(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(data[:, np.newaxis])

def adClick_predict(input_data):
    input_data_as_numpy_array = np.asarray((input_data))
    input_data_reshaped = convertedData(input_data_as_numpy_array).reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 1:
        return 'Clicked'
    else:
        return 'Not Clicked'


def main():
    st.subheader('Enter Values :')
    dailyTime = st.text_input('Enter Daily Time Spent on Site')
    age = st.text_input('Age')
    areaInc = st.text_input('Area Income')
    dailyUse = st.text_input('Daily Internet Usage')
    gender = st.text_input('Male')

    prediction = ''
    a = [dailyTime, age, areaInc, dailyUse, gender]
    if st.button('Predict The Result'):
        prediction = adClick_predict(list(map(float, a)))
        print(map(int, a))

    st.success(prediction)

    st.divider()
    st.subheader('Made By Siddharth Kishor')


if __name__ == "__main__":
    main()
