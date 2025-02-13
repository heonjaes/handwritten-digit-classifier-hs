# Use Python 3.11 as the base image
FROM python:3.11.10

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY app.py /app/
COPY requirements.txt /app/
COPY static /app/static/
COPY models /app/models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required port
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
