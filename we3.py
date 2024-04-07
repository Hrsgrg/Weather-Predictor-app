import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datetime import datetime


def load_data():
    we = pd.read_csv("output.csv")
    return we

def preprocess_data(we):
    
    we['dt_txt'] = pd.to_datetime(we["dt_txt"])
    we['date'] = we['dt_txt'].dt.day
    we['month'] = we['dt_txt'].dt.month
    we['year'] = we['dt_txt'].dt.year
    we['timezone'] = we['dt_txt'].dt.tz.utcoffset(None).total_seconds() / 3600
    we['hour'] = we['dt_txt'].dt.hour
    we['minute'] = we['dt_txt'].dt.minute
    we['second'] = we['dt_txt'].dt.second

    columns_to_encode = ['city_name', 'region', 'state']
    label_encoder = LabelEncoder()
    for col in columns_to_encode:
        we[col + '_encoded'] = label_encoder.fit_transform(we[col])
    we.drop(['dt_txt','region','state',],axis=1,inplace=True)
    X = we.drop(['main','description','city_name'], axis=1)
    y = we[['main','description']]
    return X, y, we

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300)
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def main():
    
    st.title('Weather Predictor app')

    
    st.write("This app predicts the weather based on various parameters.")
    data = load_data()
    X, y, we = preprocess_data(data)
    model = train_model(X, y)
    st.write("Model trained successfully!")
    st.write("Enter the city name to predict the weather.")
    city_name = st.text_input('Enter city name')
    temp = st.slider("Temperature (°C)", min_value=-20, max_value=60, value=20)
    temp_min=st.slider("Temperature (°C)", min_value=-20, max_value=temp+1, value=20)
    temp_max=st.slider("Temperature (°C)", min_value=temp-1, max_value=60, value=20)
    st.markdown(
    f"""
    <style>
        .st-bc {{
            background: linear-gradient(to right, #ff6666 0%, #ff6666 {100 * (temp + 20) / 60}%, #f8f9fa {100 * (temp + 20) / 60}%, #f8f9fa 100%);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

    humidity=st.number_input("enter the humidity in g/m³")
    clouds=st.number_input("enter the clouds in okta")
    wind_speed=st.number_input("enter the wind speed in kmh")
    date = st.sidebar.date_input("Select Date")
    time = st.sidebar.time_input("Select Time")
    date_time = datetime.combine(date, time)
    date = date_time.day
    month = date_time.month
    year = date_time.year
    hour = date_time.hour
    minute = date_time.minute
    second = date_time.second
    if st.button('Predict Weather'):
        
        
        city_data = we[we['city_name'] == city_name].iloc[0]
        


        id = city_data['id']
        city_id = city_data['city_id']
        latitude = city_data['latitude']
        longitude = city_data['longitude']
        sea_level = city_data['sea_level']
        grnd_level = city_data['grnd_level']
        pressure = city_data['pressure']
        wind_degree = city_data['wind_degree']
        city_name_encoded=city_data["city_name_encoded"]
        region_encoded=city_data["region_encoded"]
        state_encoded=city_data["state_encoded"]
        timezone=city_data["timezone"]

        
        input_data = np.array([id, city_id, latitude, longitude, temp, temp_min, temp_max, pressure, sea_level, grnd_level, humidity, clouds, wind_speed, wind_degree, date, month, year, timezone,hour, minute, second,city_name_encoded,region_encoded,state_encoded]).reshape(1, -1)

    
        prediction = model.predict(input_data)
        st.write('The predicted weather is:', prediction)

if __name__ == "__main__":
    main()
    
st.markdown(
    """
    <style>
        @keyframes glow {
            0% { text-shadow: 0 0 10px #ff6666; }
            50% { text-shadow: 0 0 20px #ffcc00; }
            100% { text-shadow: 0 0 10px #ff6666; }
        }
        .glowText {
            animation: glow 2s infinite alternate;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h4 class='glowText'>made by-Harsh Garg & Adamya Agarwal with💕</h4>",
    unsafe_allow_html=True
)