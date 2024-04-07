
Weather Predictor App
This is a Streamlit web application that predicts weather based on various parameters using a machine learning model.

Overview
The Weather Predictor app allows users to input weather parameters such as temperature, humidity, cloud cover, and wind speed to predict the weather for a specific city and date. The app utilizes a RandomForestClassifier model trained on weather data to make predictions.

Usage
Input Parameters: Users can input the following parameters:

City Name: Enter the name of the city for which weather prediction is needed.
Temperature: Set the current temperature.
Minimum Temperature: Set the minimum temperature.
Maximum Temperature: Set the maximum temperature.
Humidity: Enter the humidity in g/m³.
Clouds: Enter the cloud cover in okta.
Wind Speed: Enter the wind speed in km/h.
Date and Time: Select the date and time for which weather prediction is needed.
Predict Weather: After entering the parameters, click the "Predict Weather" button to get the predicted weather.

Weather Prediction: The app will display the predicted weather, including the main weather condition and description.
Run the Streamlit app:
streamlit run app.py
Contributors
Harsh Garg
Adamya Agarwal
Acknowledgements
Data source: OpenWeatherMap
Built with Streamlit
