# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the output directory and set permissions
RUN mkdir -p static/output && chmod 777 static/output

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]