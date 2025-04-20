FROM python:3.9-slim@sha256:5e5e59f40a9eff8c5f8c7a413d96da8f8744110d1d81596573d0067de8c3174c

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
