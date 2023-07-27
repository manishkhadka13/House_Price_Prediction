import gradio as gr
import requests
import pandas as pd

# Replace with the correct endpoint URL
API_ENDPOINT = "http://localhost:8084/predict/"

# Assuming you have a DataFrame called 'data' containing the input data
# Extract unique locations from the 'location' column for the dropdown options
data=pd.read_csv('cleaned_data.csv')
unique_locations = data["location"].unique().tolist()

def house_price_prediction(location, total_sqft, bath, bhk):
    try:
        data = {
            "location": location,
            "total_sqft": total_sqft,
            "bath": bath,
            "bhk": bhk
        }
        response = requests.post(API_ENDPOINT, json=data)
        response_data = response.json()

        if "prediction" in response_data:
            prediction = response_data["prediction"]
            rounded_prediction = round(prediction, 2)
            return f"{rounded_prediction} lakhs"
        else:
            return "Error: Unable to get a prediction. Please check your inputs."

    except requests.exceptions.RequestException as req_ex:
        return f"Error: Connection error. {str(req_ex)}"
    except ValueError as val_err:
        return f"Error: Invalid response from the server. {str(val_err)}"

# Define the 'location' input as a dropdown with available options
location_input = gr.inputs.Dropdown(unique_locations, label="Location")

iface = gr.Interface(
    fn=house_price_prediction,
    inputs=[location_input, "number", "number", "number"],  # Use the dropdown as the first input
    outputs="text",
    title="House Price Prediction",
    description="Predict the price of a house based on location, total_sqft, bath, and bhk."
)

iface.launch()
