import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datetime import datetime


def load_data():
    we = pd.read_csv("C:/Users/AVITA/Desktop/sample_dataset.csv")
    
    return we

def preprocess_data(we):
    rows_to_delete = int(0.9 * len(we))

    # Randomly select rows to delete
    rows_to_delete_indices = we.sample(n=rows_to_delete).index

    we.drop(index=rows_to_delete_indices, inplace=True)
        
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
    we.drop(['dt_txt', 'region', 'state'], axis=1, inplace=True)
    X = we.drop(['main', 'description', 'city_name'], axis=1)
    y = we[['main', 'description']]
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
    temp_min=st.slider("Temperature min (°C)", min_value=-20, max_value=40, value=20)
    temp_max=st.slider("Temperature max(°C)", min_value=1, max_value=60, value=20)

    humidity=st.number_input("enter the humidity in g/m³")
    clouds=st.number_input("enter the clouds in okta")
    wind_speed=st.number_input("enter the wind speed in kmh")
    time_input = st.sidebar.time_input("Select Time")
    
    date = st.sidebar.number_input("Date", min_value=1, max_value=31)
    month = st.sidebar.number_input("Month", min_value=1, max_value=12)
    year = st.sidebar.number_input("Year", min_value=1900, max_value=2100)
    hour = time_input.hour if time_input else None
    minute = time_input.minute if time_input else None
    second = time_input.second if time_input else None
    if st.button('Predict Weather'):
        if city_name:
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
    else:
        st.write("Please enter a valid city name.")
    st.markdown(
        """
        <style>
        
        .glowText {
            animation: glowing 1.5s infinite;
            font-size: 20px;
            font-family: 'Arial Black', sans-serif;
            color: #fff;
        }
        @keyframes glowing {
            0% { text-shadow: 0 0 10px #00ffff, 0 0 20px #959bab, 0 0 30px #00ffff, 0 0 40px #eda6db, 0 0 50px #00ffff, 0 0 60px #00ffff, 0 0 70px #00ffff;}
            50% { text-shadow: none; }
            100% { text-shadow: 0 0 10px #00ffff, 0 0 20px #959bab, 0 0 30px #00ffff, 0 0 40px #eda6db, 0 0 50px #00ffff, 0 0 60px #00ffff, 0 0 70px #00ffff; }
        }
        
        </style>
        """
    "<h4 class='glowText animate__animated animate__pulse'>Made by Harsh Garg & Adamya Agarwal with 💕</h4>",
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
    


