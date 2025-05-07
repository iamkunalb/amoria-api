# Use a minimal Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run expects
EXPOSE 8080

# Launch FastAPI app on port 8080
CMD ["uvicorn", "matchmaker_api:app", "--host", "0.0.0.0", "--port", "8080"]
