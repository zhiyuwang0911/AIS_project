# # Use an official Python image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy application files
# COPY . /app

# # Install required Python packages
# RUN pip install --no-cache-dir -r requirement.txt

# # Expose the port that Streamlit runs on
# EXPOSE 8502

# # Command to run the Streamlit application
# CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.enableCORS=false"]

FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports for Streamlit and Prometheus
EXPOSE 8501
EXPOSE 8000

# Run the Streamlit app
CMD ["streamlit", "run", "app5.py", "--server.port=8501", "--server.enableCORS=false"]
