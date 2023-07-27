# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all the contents of the current directory into the container
COPY . .

# Install the required dependencies
RUN pip install -r requirements.txt

# Expose the port that gradio will be running
EXPOSE 7860

# Set the command to run your gradio app
CMD ["python", "main.py"]