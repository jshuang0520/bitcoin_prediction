FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create utilities directory
RUN mkdir -p /app/utilities

# Copy configs and utilities
COPY configs /app/configs
COPY utilities /app/utilities

# Copy the web application code
COPY web_app .

# Create data directories
RUN mkdir -p /app/data/raw /app/data/predictions

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "backend/app.py"] 