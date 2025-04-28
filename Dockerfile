FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/models logs

# Set environment variables
ENV FLASK_ENV=production
# The SECRET_KEY will be generated at runtime in the production.py file

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "run.py", "web", "--production", "--host", "0.0.0.0", "--port", "5000"]
