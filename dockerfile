# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application, including the dataset
COPY . .

# --- Pre-build the knowledge base ---
# This is a key step. It runs the build_database.py script when the image is built.
# This means the final container is instantly ready to use without any setup.
RUN python build_database.py

# The command to run the chatbot when the container starts
CMD ["python", "mediBot_cli.py"]