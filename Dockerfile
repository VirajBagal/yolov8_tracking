# Base image with Python
FROM python:3.9-slim

# Install FFmpeg
RUN apt-get update && apt-get install build-essential -y && apt-get install manpages-dev -y && apt-get install -y ffmpeg

# Set the working directory
WORKDIR /app

# Copy only the requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies, if any
RUN pip install -r requirements.txt

# required for tracking
RUN pip install lap==0.4.0

# Copy all files
COPY . /app

#expose port 80
EXPOSE 80

# Specify the default command to run when the container starts
CMD ["streamlit", "run", "app.py", "--server.port", "80", "--server.address", "0.0.0.0"]