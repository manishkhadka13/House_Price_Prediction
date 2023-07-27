# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all the contents of the current directory into the container
COPY . .

# Install the required dependencies
RUN pip install --upgrade pip -r requirements.txt

# Expose the port that Streamlit will be running on (replace 8501 with your desired port)
EXPOSE 8501

# Set the command to run your Streamlit app
ENTRYPOINT ["gradio", "run"]

CMD ["main.py"]