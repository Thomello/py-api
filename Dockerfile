FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install -r requirements.txt

# Expose the port that the application will run on
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]
