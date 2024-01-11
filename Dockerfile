# Use an NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install the libsndfile library
RUN apt-get install -y libsndfile1

# Set the working directory
WORKDIR /app

# Copy the source code
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]