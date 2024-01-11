# Use an NVIDIA CUDA base image with Python
FROM alexthesun/cuda116-majbyrapi-base:v0.1

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container
COPY . .

RUN pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

